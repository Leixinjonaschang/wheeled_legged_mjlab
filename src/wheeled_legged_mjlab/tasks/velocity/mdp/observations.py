from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor, RayCastSensor
from mjlab.sensor.camera_sensor import CameraSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.utils.lab_api.math import quat_apply_inverse
from torchvision.transforms.functional import gaussian_blur

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_height(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Per-foot vertical clearance above terrain.

  Returns:
    Tensor of shape [B, F] where F is the number of frames (feet).
  """
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, TerrainHeightSensor), (
    f"foot_height requires a TerrainHeightSensor, got {type(sensor).__name__}"
  )
  heights = sensor.data.heights
  if heights.ndim == 3:
    return heights.amin(dim=-1)
  if heights.ndim == 2:
    return heights
  raise ValueError(
    "foot_height expects terrain clearance samples with shape [B, F] or "
    f"[B, F, N], got {tuple(heights.shape)}"
  )


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Contact forces expressed in the robot body frame."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_w = torch.nan_to_num(sensor_data.force, nan=0.0, posinf=0.0, neginf=0.0)
  asset = env.scene[asset_cfg.name]
  root_quat_w = asset.data.root_link_quat_w[:, None, :].expand(
    -1, forces_w.shape[1], -1
  )
  forces_b = quat_apply_inverse(root_quat_w, forces_w)
  forces_flat = forces_b.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def _validate_depth_range(depth_min_m: float, depth_max_m: float) -> None:
  if depth_min_m < 0.0:
    raise ValueError(f"depth_min_m must be >= 0, got {depth_min_m}")
  if depth_max_m <= depth_min_m:
    raise ValueError(
      f"depth_max_m must be greater than depth_min_m, got {depth_max_m}"
    )


def _crop_depth_image(depth_m: torch.Tensor, left_crop: int) -> torch.Tensor:
  if depth_m.ndim < 2:
    raise ValueError(
      "depth_m must have at least two dimensions ending in [H, W], "
      f"got shape {tuple(depth_m.shape)}"
    )
  if left_crop < 0 or left_crop >= depth_m.shape[-1]:
    raise ValueError(
      f"left_crop must be in [0, {depth_m.shape[-1] - 1}], got {left_crop}"
    )
  return depth_m[..., left_crop:].to(dtype=torch.float32).contiguous()


def _encode_invalid_depth(depth_m: torch.Tensor) -> torch.Tensor:
  """Encode every invalid depth sample as the finite sentinel 0 m."""
  invalid = (~torch.isfinite(depth_m)) | (depth_m <= 0.0)
  return torch.where(invalid, torch.zeros_like(depth_m), depth_m).contiguous()


def preprocess_depth_image(
  depth_m: torch.Tensor,
  *,
  left_crop: int = 0,
) -> torch.Tensor:
  """Crop metric depth and encode invalid samples as 0 m."""
  depth_m = _crop_depth_image(depth_m, left_crop)
  return _encode_invalid_depth(depth_m)


def _camera_depth_image_meters(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  camera: CameraSensor = env.scene[sensor_name]
  assert camera.data.depth is not None, f"Sensor '{sensor_name}' has no depth data"
  return camera.data.depth.squeeze(-1)


def depth_image(
  env: ManagerBasedRlEnv,
  sensor_name: str = "depth_camera",
  left_crop: int = 0,
) -> torch.Tensor:
  """Finite metric depth image from the forward-facing camera."""
  return preprocess_depth_image(
    _camera_depth_image_meters(env, sensor_name),
    left_crop=left_crop,
  )


class _DepthFrameProcessor:
  """Apply capture-time depth randomization followed by deterministic processing."""

  def __init__(self) -> None:
    self._calibration_scale: torch.Tensor | None = None
    self._calibration_bias_m: torch.Tensor | None = None
    self._calibration_scale_range: tuple[float, float] | None = None
    self._calibration_bias_range_m: tuple[float, float] | None = None
    self._randomization_enabled = False

  def __call__(
    self,
    depth_m: torch.Tensor,
    *,
    env_ids: torch.Tensor | None = None,
    left_crop: int = 0,
    depth_min_m: float = 0.2,
    depth_max_m: float = 2.0,
    enable_depth_randomization: bool = False,
    calibration_scale_range: tuple[float, float] = (1.0, 1.0),
    calibration_bias_range_m: tuple[float, float] = (0.0, 0.0),
    enable_depth_distance_noise: bool = True,
    enable_depth_gaussian_blur: bool = False,
    enable_depth_edge_noise: bool = False,
    enable_depth_dropout: bool = False,
    noise_base_m: float = 0.001,
    noise_quadratic_coeff: float = 0.005,
    gaussian_blur_kernel_size: tuple[int, int] = (3, 3),
    gaussian_blur_sigma: float = 1.0,
    edge_focal_length_px: float = 28.0,
    edge_baseline_m: float = 0.05,
    edge_disparity_threshold_px: float = 0.5,
    edge_band_radius_px: int = 1,
    edge_corruption_probability: float = 0.5,
    edge_empty_ratio: float = 0.5,
    dropout_probability: float = 0.0,
    dropout_patch_count_range: tuple[int, int] = (1, 1),
    dropout_area_fraction_range: tuple[float, float] = (0.0, 0.0),
    dropout_aspect_ratio_range: tuple[float, float] = (1.0, 1.0),
  ) -> torch.Tensor:
    _validate_depth_range(depth_min_m, depth_max_m)
    self._validate_randomization_cfg(
      calibration_scale_range=calibration_scale_range,
      calibration_bias_range_m=calibration_bias_range_m,
      noise_base_m=noise_base_m,
      noise_quadratic_coeff=noise_quadratic_coeff,
      gaussian_blur_kernel_size=gaussian_blur_kernel_size,
      gaussian_blur_sigma=gaussian_blur_sigma,
      edge_focal_length_px=edge_focal_length_px,
      edge_baseline_m=edge_baseline_m,
      edge_disparity_threshold_px=edge_disparity_threshold_px,
      edge_band_radius_px=edge_band_radius_px,
      edge_corruption_probability=edge_corruption_probability,
      edge_empty_ratio=edge_empty_ratio,
      dropout_probability=dropout_probability,
      dropout_patch_count_range=dropout_patch_count_range,
      dropout_area_fraction_range=dropout_area_fraction_range,
      dropout_aspect_ratio_range=dropout_aspect_ratio_range,
    )
    depth_m = _crop_depth_image(depth_m, left_crop)
    if depth_m.ndim != 3:
      raise ValueError(
        "buffered depth_m must have shape [B, H, W], "
        f"got {tuple(depth_m.shape)}"
      )
    original_invalid = (~torch.isfinite(depth_m)) | (depth_m <= 0.0)

    self._randomization_enabled = enable_depth_randomization
    if env_ids is not None:
      env_ids = env_ids.to(device=depth_m.device, dtype=torch.long)

    if enable_depth_randomization:
      self._ensure_calibration(
        batch_size=depth_m.shape[0],
        device=depth_m.device,
        calibration_scale_range=calibration_scale_range,
        calibration_bias_range_m=calibration_bias_range_m,
      )
      assert self._calibration_scale is not None
      assert self._calibration_bias_m is not None
      if env_ids is None:
        scale = self._calibration_scale
        bias_m = self._calibration_bias_m
      else:
        depth_m = depth_m.index_select(0, env_ids)
        original_invalid = original_invalid.index_select(0, env_ids)
        scale = self._calibration_scale.index_select(0, env_ids)
        bias_m = self._calibration_bias_m.index_select(0, env_ids)
      edge_reference_depth_m = depth_m
      edge_reference_invalid = original_invalid
      depth_m = depth_m * scale + bias_m
      if enable_depth_distance_noise:
        depth_for_sigma = torch.nan_to_num(
          depth_m,
          nan=depth_max_m,
          posinf=depth_max_m,
          neginf=depth_min_m,
        ).clamp(min=depth_min_m, max=depth_max_m)
        sigma = noise_base_m + noise_quadratic_coeff * depth_for_sigma.square()
        depth_m = depth_m + sigma * torch.randn_like(depth_m)
      if enable_depth_gaussian_blur:
        depth_for_blur = torch.nan_to_num(
          depth_m,
          nan=depth_max_m,
          posinf=depth_max_m,
          neginf=depth_max_m,
        ).masked_fill(original_invalid, depth_max_m)
        depth_for_blur = torch.where(
          depth_for_blur >= depth_min_m,
          depth_for_blur,
          torch.full_like(depth_for_blur, depth_max_m),
        ).clamp(max=depth_max_m)
        depth_m = gaussian_blur(
          depth_for_blur.unsqueeze(1),
          kernel_size=list(gaussian_blur_kernel_size),
          sigma=[gaussian_blur_sigma, gaussian_blur_sigma],
        ).squeeze(1)
      if enable_depth_edge_noise:
        depth_m = self._apply_depth_edge_noise(
          depth_m,
          reference_depth_m=edge_reference_depth_m,
          reference_invalid=edge_reference_invalid,
          depth_min_m=depth_min_m,
          depth_max_m=depth_max_m,
          focal_length_px=edge_focal_length_px,
          baseline_m=edge_baseline_m,
          disparity_threshold_px=edge_disparity_threshold_px,
          band_radius_px=edge_band_radius_px,
          corruption_probability=edge_corruption_probability,
          empty_ratio=edge_empty_ratio,
        )
      if enable_depth_dropout and dropout_probability > 0.0:
        dropout_mask = self._structured_dropout_mask(
          batch_size=depth_m.shape[0],
          height=depth_m.shape[-2],
          width=depth_m.shape[-1],
          device=depth_m.device,
          probability=dropout_probability,
          patch_count_range=dropout_patch_count_range,
          area_fraction_range=dropout_area_fraction_range,
          aspect_ratio_range=dropout_aspect_ratio_range,
        )
        depth_m = depth_m.masked_fill(dropout_mask, torch.nan)
    elif env_ids is not None:
      depth_m = depth_m.index_select(0, env_ids)
      original_invalid = original_invalid.index_select(0, env_ids)

    depth_m = depth_m.masked_fill(original_invalid, 0.0)
    return _encode_invalid_depth(depth_m)

  @staticmethod
  def _apply_depth_edge_noise(
    depth_m: torch.Tensor,
    *,
    reference_depth_m: torch.Tensor,
    reference_invalid: torch.Tensor,
    depth_min_m: float,
    depth_max_m: float,
    focal_length_px: float,
    baseline_m: float,
    disparity_threshold_px: float,
    band_radius_px: int,
    corruption_probability: float,
    empty_ratio: float,
  ) -> torch.Tensor:
    if corruption_probability <= 0.0:
      return depth_m

    valid = ~reference_invalid
    safe_min_m = max(depth_min_m, torch.finfo(reference_depth_m.dtype).eps)
    reference_depth_m = torch.nan_to_num(
      reference_depth_m,
      nan=depth_max_m,
      posinf=depth_max_m,
      neginf=safe_min_m,
    ).clamp(min=safe_min_m, max=depth_max_m)
    disparity = focal_length_px * baseline_m / reference_depth_m

    edge_band = torch.zeros_like(valid)
    horizontal_edge = (
      valid[..., :, :-1]
      & valid[..., :, 1:]
      & (
        torch.abs(disparity[..., :, :-1] - disparity[..., :, 1:])
        > disparity_threshold_px
      )
    )
    edge_band[..., :, :-1] |= horizontal_edge
    edge_band[..., :, 1:] |= horizontal_edge
    vertical_edge = (
      valid[..., :-1, :]
      & valid[..., 1:, :]
      & (
        torch.abs(disparity[..., :-1, :] - disparity[..., 1:, :])
        > disparity_threshold_px
      )
    )
    edge_band[..., :-1, :] |= vertical_edge
    edge_band[..., 1:, :] |= vertical_edge

    for _ in range(band_radius_px - 1):
      previous_band = edge_band
      expanded_band = previous_band.clone()
      expanded_band[..., 1:, :] |= previous_band[..., :-1, :]
      expanded_band[..., :-1, :] |= previous_band[..., 1:, :]
      expanded_band[..., :, 1:] |= previous_band[..., :, :-1]
      expanded_band[..., :, :-1] |= previous_band[..., :, 1:]
      edge_band = expanded_band

    corrupt_mask = edge_band & (
      torch.rand_like(depth_m) < corruption_probability
    )
    empty_mask = corrupt_mask & (torch.rand_like(depth_m) < empty_ratio)
    replace_mask = corrupt_mask & ~empty_mask

    replacement = depth_m
    best_score = torch.full_like(depth_m, -1.0)
    height, width = depth_m.shape[-2:]
    for distance in range(1, band_radius_px + 1):
      for row_offset, col_offset in (
        (-distance, 0),
        (distance, 0),
        (0, -distance),
        (0, distance),
      ):
        shifts = (-row_offset, -col_offset)
        candidate_disparity = torch.roll(
          disparity,
          shifts=shifts,
          dims=(-2, -1),
        )
        candidate_valid = torch.roll(valid, shifts=shifts, dims=(-2, -1))
        candidate_depth = torch.roll(depth_m, shifts=shifts, dims=(-2, -1))
        in_bounds = torch.ones_like(valid)
        if row_offset < 0:
          in_bounds[..., : -row_offset, :] = False
        elif row_offset > 0:
          in_bounds[..., height - row_offset :, :] = False
        if col_offset < 0:
          in_bounds[..., :, : -col_offset] = False
        elif col_offset > 0:
          in_bounds[..., :, width - col_offset :] = False

        is_cross_edge_neighbor = (
          replace_mask
          & valid
          & candidate_valid
          & in_bounds
          & (
            torch.abs(disparity - candidate_disparity)
            > disparity_threshold_px
          )
        )
        score = torch.rand_like(depth_m)
        take_candidate = is_cross_edge_neighbor & (score > best_score)
        replacement = torch.where(take_candidate, candidate_depth, replacement)
        best_score = torch.where(take_candidate, score, best_score)

    depth_m = torch.where(empty_mask, torch.zeros_like(depth_m), depth_m)
    return torch.where(replace_mask & (best_score >= 0.0), replacement, depth_m)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if not self._randomization_enabled or self._calibration_scale is None:
      return
    assert self._calibration_bias_m is not None
    assert self._calibration_scale_range is not None
    assert self._calibration_bias_range_m is not None
    if env_ids is None or isinstance(env_ids, slice):
      env_ids = torch.arange(
        self._calibration_scale.shape[0],
        device=self._calibration_scale.device,
      )
    else:
      env_ids = env_ids.to(device=self._calibration_scale.device, dtype=torch.long)
    if env_ids.numel() == 0:
      return
    self._sample_calibration(env_ids)

  def _ensure_calibration(
    self,
    *,
    batch_size: int,
    device: torch.device,
    calibration_scale_range: tuple[float, float],
    calibration_bias_range_m: tuple[float, float],
  ) -> None:
    ranges_changed = (
      self._calibration_scale_range != calibration_scale_range
      or self._calibration_bias_range_m != calibration_bias_range_m
    )
    needs_init = (
      self._calibration_scale is None
      or self._calibration_scale.shape[0] != batch_size
      or self._calibration_scale.device != device
      or ranges_changed
    )
    if not needs_init:
      return
    self._calibration_scale_range = calibration_scale_range
    self._calibration_bias_range_m = calibration_bias_range_m
    self._calibration_scale = torch.empty(
      batch_size, 1, 1, device=device, dtype=torch.float32
    )
    self._calibration_bias_m = torch.empty_like(self._calibration_scale)
    self._sample_calibration(torch.arange(batch_size, device=device))

  def _sample_calibration(self, env_ids: torch.Tensor) -> None:
    assert self._calibration_scale is not None
    assert self._calibration_bias_m is not None
    assert self._calibration_scale_range is not None
    assert self._calibration_bias_range_m is not None
    count = env_ids.numel()
    scale_min, scale_max = self._calibration_scale_range
    bias_min_m, bias_max_m = self._calibration_bias_range_m
    if scale_min == scale_max:
      self._calibration_scale[env_ids] = scale_min
    else:
      self._calibration_scale[env_ids] = scale_min + (
        scale_max - scale_min
      ) * torch.rand(
        count,
        1,
        1,
        device=self._calibration_scale.device,
      )
    if bias_min_m == bias_max_m:
      self._calibration_bias_m[env_ids] = bias_min_m
    else:
      self._calibration_bias_m[env_ids] = bias_min_m + (
        bias_max_m - bias_min_m
      ) * torch.rand(
        count,
        1,
        1,
        device=self._calibration_bias_m.device,
      )

  @staticmethod
  def _structured_dropout_mask(
    *,
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    probability: float,
    patch_count_range: tuple[int, int],
    area_fraction_range: tuple[float, float],
    aspect_ratio_range: tuple[float, float],
  ) -> torch.Tensor:
    mask = torch.zeros(batch_size, height, width, device=device, dtype=torch.bool)
    if probability <= 0.0 or area_fraction_range[1] <= 0.0:
      return mask

    min_patches, max_patches = patch_count_range
    pixel_count = height * width
    min_area = max(
      min_patches,
      math.ceil(area_fraction_range[0] * pixel_count),
    )
    max_area = max(1, math.floor(area_fraction_range[1] * pixel_count))
    if max_area < min_area:
      raise ValueError(
        "dropout_area_fraction_range does not contain a whole-pixel area for "
        f"an image of shape ({height}, {width})"
      )
    if max_area < min_patches:
      raise ValueError(
        "dropout area budget must contain at least one pixel per minimum patch"
      )

    frame_active = torch.rand(batch_size, device=device) < probability
    patch_counts = torch.randint(
      min_patches,
      max_patches + 1,
      (batch_size,),
      device=device,
    )
    area_budgets = torch.randint(
      min_area,
      max_area + 1,
      (batch_size,),
      device=device,
    )
    patch_counts = torch.where(
      frame_active,
      patch_counts,
      torch.zeros_like(patch_counts),
    )

    slots = torch.arange(max_patches, device=device).unsqueeze(0)
    active_slots = slots < patch_counts.unsqueeze(1)
    safe_counts = patch_counts.clamp_min(1)
    base_areas = area_budgets // safe_counts
    remainders = area_budgets % safe_counts
    patch_areas = base_areas.unsqueeze(1) + (slots < remainders.unsqueeze(1))
    patch_areas = torch.where(
      active_slots,
      patch_areas,
      torch.zeros_like(patch_areas),
    )

    aspect_min, aspect_max = aspect_ratio_range
    log_aspect = math.log(aspect_min) + (
      math.log(aspect_max) - math.log(aspect_min)
    ) * torch.rand(batch_size, max_patches, device=device)
    aspect = torch.exp(log_aspect)
    safe_areas = patch_areas.clamp_min(1)
    patch_heights = torch.floor(
      torch.sqrt(safe_areas.to(torch.float32) / aspect)
    ).to(torch.long)
    patch_heights = patch_heights.clamp(min=1, max=height)
    patch_heights = torch.minimum(patch_heights, safe_areas)
    patch_widths = torch.floor(
      torch.sqrt(safe_areas.to(torch.float32) * aspect)
    ).to(torch.long)
    patch_widths = patch_widths.clamp(min=1, max=width)
    patch_widths = torch.minimum(
      patch_widths,
      torch.div(safe_areas, patch_heights, rounding_mode="floor").clamp_min(1),
    )

    top = torch.floor(
      torch.rand(batch_size, max_patches, device=device)
      * (height - patch_heights + 1)
    ).to(torch.long)
    left = torch.floor(
      torch.rand(batch_size, max_patches, device=device)
      * (width - patch_widths + 1)
    ).to(torch.long)
    rows = torch.arange(height, device=device).view(1, 1, height, 1)
    cols = torch.arange(width, device=device).view(1, 1, 1, width)
    rectangles = (
      active_slots[:, :, None, None]
      & (rows >= top[:, :, None, None])
      & (rows < (top + patch_heights)[:, :, None, None])
      & (cols >= left[:, :, None, None])
      & (cols < (left + patch_widths)[:, :, None, None])
    )
    return rectangles.any(dim=1)

  @staticmethod
  def _validate_randomization_cfg(
    *,
    calibration_scale_range: tuple[float, float],
    calibration_bias_range_m: tuple[float, float],
    noise_base_m: float,
    noise_quadratic_coeff: float,
    gaussian_blur_kernel_size: tuple[int, int],
    gaussian_blur_sigma: float,
    edge_focal_length_px: float,
    edge_baseline_m: float,
    edge_disparity_threshold_px: float,
    edge_band_radius_px: int,
    edge_corruption_probability: float,
    edge_empty_ratio: float,
    dropout_probability: float,
    dropout_patch_count_range: tuple[int, int],
    dropout_area_fraction_range: tuple[float, float],
    dropout_aspect_ratio_range: tuple[float, float],
  ) -> None:
    scale_min, scale_max = calibration_scale_range
    if scale_min <= 0.0 or scale_min > scale_max:
      raise ValueError(
        "calibration_scale_range must be positive and ordered min <= max"
      )
    bias_min_m, bias_max_m = calibration_bias_range_m
    if bias_min_m > bias_max_m:
      raise ValueError(
        "calibration_bias_range_m must be ordered min <= max"
      )
    if not math.isfinite(noise_base_m) or noise_base_m < 0.0:
      raise ValueError(
        f"noise_base_m must be finite and non-negative, got {noise_base_m}"
      )
    if not math.isfinite(noise_quadratic_coeff) or noise_quadratic_coeff < 0.0:
      raise ValueError(
        "noise_quadratic_coeff must be finite and non-negative, "
        f"got {noise_quadratic_coeff}"
      )
    if len(gaussian_blur_kernel_size) != 2 or any(
      size <= 0 or size % 2 == 0 for size in gaussian_blur_kernel_size
    ):
      raise ValueError(
        "gaussian_blur_kernel_size must contain two positive odd values, "
        f"got {gaussian_blur_kernel_size}"
      )
    if not math.isfinite(gaussian_blur_sigma) or gaussian_blur_sigma <= 0.0:
      raise ValueError(
        "gaussian_blur_sigma must be finite and positive, "
        f"got {gaussian_blur_sigma}"
      )
    if not math.isfinite(edge_focal_length_px) or edge_focal_length_px <= 0.0:
      raise ValueError(
        "edge_focal_length_px must be finite and positive, "
        f"got {edge_focal_length_px}"
      )
    if not math.isfinite(edge_baseline_m) or edge_baseline_m <= 0.0:
      raise ValueError(
        f"edge_baseline_m must be finite and positive, got {edge_baseline_m}"
      )
    if (
      not math.isfinite(edge_disparity_threshold_px)
      or edge_disparity_threshold_px <= 0.0
    ):
      raise ValueError(
        "edge_disparity_threshold_px must be finite and positive, "
        f"got {edge_disparity_threshold_px}"
      )
    if edge_band_radius_px not in (1, 2):
      raise ValueError(
        f"edge_band_radius_px must be 1 or 2, got {edge_band_radius_px}"
      )
    if not 0.0 <= edge_corruption_probability <= 1.0:
      raise ValueError(
        "edge_corruption_probability must be in [0, 1], "
        f"got {edge_corruption_probability}"
      )
    if not 0.0 <= edge_empty_ratio <= 1.0:
      raise ValueError(
        f"edge_empty_ratio must be in [0, 1], got {edge_empty_ratio}"
      )
    if not 0.0 <= dropout_probability <= 1.0:
      raise ValueError(
        "dropout_probability must be in [0, 1], "
        f"got {dropout_probability}"
      )
    patch_min, patch_max = dropout_patch_count_range
    if patch_min < 1 or patch_min > patch_max:
      raise ValueError(
        "dropout_patch_count_range must be positive and ordered min <= max"
      )
    area_min, area_max = dropout_area_fraction_range
    if area_min < 0.0 or area_min > area_max or area_max > 1.0:
      raise ValueError(
        "dropout_area_fraction_range must be ordered within [0, 1]"
      )
    aspect_min, aspect_max = dropout_aspect_ratio_range
    if aspect_min <= 0.0 or aspect_min > aspect_max:
      raise ValueError(
        "dropout_aspect_ratio_range must be positive and ordered min <= max"
      )


class DepthBuffer:
  """Depth image buffer updated at a lower policy-step rate."""

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    del cfg, env
    self._processor = _DepthFrameProcessor()
    self._buffer: torch.Tensor | None = None
    self._last_update_step: int | None = None
    self._invalid_env_ids: torch.Tensor | None = None

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str = "depth_camera",
    buffer_size: int = 5,
    update_period: int = 5,
    left_crop: int = 0,
    depth_min_m: float = 0.2,
    depth_max_m: float = 2.0,
    enable_depth_randomization: bool = False,
    calibration_scale_range: tuple[float, float] = (1.0, 1.0),
    calibration_bias_range_m: tuple[float, float] = (0.0, 0.0),
    enable_depth_distance_noise: bool = True,
    enable_depth_gaussian_blur: bool = False,
    enable_depth_edge_noise: bool = False,
    enable_depth_dropout: bool = False,
    noise_base_m: float = 0.001,
    noise_quadratic_coeff: float = 0.005,
    gaussian_blur_kernel_size: tuple[int, int] = (3, 3),
    gaussian_blur_sigma: float = 1.0,
    edge_focal_length_px: float = 28.0,
    edge_baseline_m: float = 0.05,
    edge_disparity_threshold_px: float = 0.5,
    edge_band_radius_px: int = 1,
    edge_corruption_probability: float = 0.5,
    edge_empty_ratio: float = 0.5,
    dropout_probability: float = 0.0,
    dropout_patch_count_range: tuple[int, int] = (1, 1),
    dropout_area_fraction_range: tuple[float, float] = (0.0, 0.0),
    dropout_aspect_ratio_range: tuple[float, float] = (1.0, 1.0),
  ) -> torch.Tensor:
    if buffer_size < 1:
      raise ValueError(f"buffer_size must be >= 1, got {buffer_size}")
    if update_period < 1:
      raise ValueError(f"update_period must be >= 1, got {update_period}")

    step = int(getattr(env, "common_step_counter", 0))
    needs_init = self._buffer is None or self._buffer.shape[1] != buffer_size
    needs_reset_fill = self._invalid_env_ids is not None
    needs_periodic_update = (
      self._last_update_step is None
      or step - self._last_update_step >= update_period
    )

    if not (needs_init or needs_reset_fill or needs_periodic_update):
      assert self._buffer is not None
      return self._buffer

    reset_env_ids = None
    if needs_reset_fill:
      assert self._invalid_env_ids is not None
      reset_env_ids = self._invalid_env_ids.to(dtype=torch.long)
    process_env_ids = (
      None if needs_init or needs_periodic_update else reset_env_ids
    )
    frame = self._processor(
      _camera_depth_image_meters(env, sensor_name),
      env_ids=process_env_ids,
      left_crop=left_crop,
      depth_min_m=depth_min_m,
      depth_max_m=depth_max_m,
      enable_depth_randomization=enable_depth_randomization,
      calibration_scale_range=calibration_scale_range,
      calibration_bias_range_m=calibration_bias_range_m,
      enable_depth_distance_noise=enable_depth_distance_noise,
      enable_depth_gaussian_blur=enable_depth_gaussian_blur,
      enable_depth_edge_noise=enable_depth_edge_noise,
      enable_depth_dropout=enable_depth_dropout,
      noise_base_m=noise_base_m,
      noise_quadratic_coeff=noise_quadratic_coeff,
      gaussian_blur_kernel_size=gaussian_blur_kernel_size,
      gaussian_blur_sigma=gaussian_blur_sigma,
      edge_focal_length_px=edge_focal_length_px,
      edge_baseline_m=edge_baseline_m,
      edge_disparity_threshold_px=edge_disparity_threshold_px,
      edge_band_radius_px=edge_band_radius_px,
      edge_corruption_probability=edge_corruption_probability,
      edge_empty_ratio=edge_empty_ratio,
      dropout_probability=dropout_probability,
      dropout_patch_count_range=dropout_patch_count_range,
      dropout_area_fraction_range=dropout_area_fraction_range,
      dropout_aspect_ratio_range=dropout_aspect_ratio_range,
    )

    if needs_init:
      self._buffer = frame.unsqueeze(1).repeat(
        1, buffer_size, *(1 for _ in frame.shape[1:])
      )
      self._last_update_step = step
      self._invalid_env_ids = None
      return self._buffer

    if reset_env_ids is not None:
      assert self._buffer is not None
      reset_env_ids = reset_env_ids.to(device=frame.device)
      reset_frame = (
        frame.index_select(0, reset_env_ids)
        if process_env_ids is None
        else frame
      )
      self._buffer[reset_env_ids] = reset_frame.unsqueeze(1).expand(
        -1, buffer_size, *reset_frame.shape[1:]
      )
      self._invalid_env_ids = None

    if needs_periodic_update:
      assert self._buffer is not None
      self._buffer = torch.roll(self._buffer, shifts=-1, dims=1)
      self._buffer[:, -1] = frame
      self._last_update_step = step

    assert self._buffer is not None
    return self._buffer

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._processor.reset(env_ids)
    if env_ids is None or isinstance(env_ids, slice):
      self._buffer = None
      self._last_update_step = None
      self._invalid_env_ids = None
      return
    if self._buffer is None or env_ids.numel() == 0:
      return
    env_ids = env_ids.to(device=self._buffer.device, dtype=torch.long)
    if self._invalid_env_ids is None:
      self._invalid_env_ids = env_ids
    else:
      self._invalid_env_ids = torch.unique(
        torch.cat((self._invalid_env_ids, env_ids))
      )


depth_buffer = DepthBuffer


class AsyncDepthBuffer:
  """Depth frames captured on a nominal clock with per-environment delay."""

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    del cfg, env
    self._processor = _DepthFrameProcessor()
    self._frames: torch.Tensor | None = None
    self._capture_times_s: torch.Tensor | None = None
    self._next_capture_time_s: float | None = None
    self._capture_period_s: float | None = None
    self._delay_s: torch.Tensor | None = None
    self._delay_range_s: tuple[float, float] | None = None
    self._invalid_env_ids: torch.Tensor | None = None

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str = "depth_camera",
    capture_frequency_hz: float = 30.0,
    system_delay_range_s: tuple[float, float] = (0.0, 0.0),
    left_crop: int = 0,
    depth_min_m: float = 0.2,
    depth_max_m: float = 2.0,
    enable_depth_randomization: bool = False,
    calibration_scale_range: tuple[float, float] = (1.0, 1.0),
    calibration_bias_range_m: tuple[float, float] = (0.0, 0.0),
    enable_depth_distance_noise: bool = True,
    enable_depth_gaussian_blur: bool = False,
    enable_depth_edge_noise: bool = False,
    enable_depth_dropout: bool = False,
    noise_base_m: float = 0.001,
    noise_quadratic_coeff: float = 0.005,
    gaussian_blur_kernel_size: tuple[int, int] = (3, 3),
    gaussian_blur_sigma: float = 1.0,
    edge_focal_length_px: float = 28.0,
    edge_baseline_m: float = 0.05,
    edge_disparity_threshold_px: float = 0.5,
    edge_band_radius_px: int = 1,
    edge_corruption_probability: float = 0.5,
    edge_empty_ratio: float = 0.5,
    dropout_probability: float = 0.0,
    dropout_patch_count_range: tuple[int, int] = (1, 1),
    dropout_area_fraction_range: tuple[float, float] = (0.0, 0.0),
    dropout_aspect_ratio_range: tuple[float, float] = (1.0, 1.0),
  ) -> torch.Tensor:
    if capture_frequency_hz <= 0.0:
      raise ValueError(
        f"capture_frequency_hz must be positive, got {capture_frequency_hz}"
      )
    step_dt = float(env.step_dt)
    if step_dt <= 0.0:
      raise ValueError(f"env.step_dt must be positive, got {step_dt}")

    delay_min_s = float(system_delay_range_s[0])
    delay_max_s = float(system_delay_range_s[1])
    if delay_min_s < 0.0:
      raise ValueError(
        f"system_delay_range_s must be non-negative, got {system_delay_range_s}"
      )
    if delay_min_s > delay_max_s:
      raise ValueError(
        "system_delay_range_s must be ordered min <= max, "
        f"got {system_delay_range_s}"
      )

    current_time_s = int(getattr(env, "common_step_counter", 0)) * step_dt
    capture_period_s = 1.0 / capture_frequency_hz
    delay_range_s = (delay_min_s, delay_max_s)
    history_size = max(1, math.ceil(delay_max_s / capture_period_s) + 1)
    config_changed = (
      self._capture_period_s != capture_period_s
      or self._delay_range_s != delay_range_s
    )
    needs_init = (
      self._frames is None
      or self._frames.shape[0] != history_size
      or config_changed
    )
    capture_due = (
      self._next_capture_time_s is None
      or current_time_s + 1.0e-9 >= self._next_capture_time_s
    )
    needs_reset_fill = self._invalid_env_ids is not None

    if not (needs_init or capture_due or needs_reset_fill):
      return self._select(current_time_s)

    reset_env_ids = None
    if needs_reset_fill:
      assert self._invalid_env_ids is not None
      reset_env_ids = self._invalid_env_ids.to(dtype=torch.long)
    process_env_ids = None if needs_init or capture_due else reset_env_ids
    frame = self._processor(
      _camera_depth_image_meters(env, sensor_name),
      env_ids=process_env_ids,
      left_crop=left_crop,
      depth_min_m=depth_min_m,
      depth_max_m=depth_max_m,
      enable_depth_randomization=enable_depth_randomization,
      calibration_scale_range=calibration_scale_range,
      calibration_bias_range_m=calibration_bias_range_m,
      enable_depth_distance_noise=enable_depth_distance_noise,
      enable_depth_gaussian_blur=enable_depth_gaussian_blur,
      enable_depth_edge_noise=enable_depth_edge_noise,
      enable_depth_dropout=enable_depth_dropout,
      noise_base_m=noise_base_m,
      noise_quadratic_coeff=noise_quadratic_coeff,
      gaussian_blur_kernel_size=gaussian_blur_kernel_size,
      gaussian_blur_sigma=gaussian_blur_sigma,
      edge_focal_length_px=edge_focal_length_px,
      edge_baseline_m=edge_baseline_m,
      edge_disparity_threshold_px=edge_disparity_threshold_px,
      edge_band_radius_px=edge_band_radius_px,
      edge_corruption_probability=edge_corruption_probability,
      edge_empty_ratio=edge_empty_ratio,
      dropout_probability=dropout_probability,
      dropout_patch_count_range=dropout_patch_count_range,
      dropout_area_fraction_range=dropout_area_fraction_range,
      dropout_aspect_ratio_range=dropout_aspect_ratio_range,
    ).unsqueeze(1)

    if needs_init:
      self._capture_period_s = capture_period_s
      self._delay_range_s = delay_range_s
      self._frames = frame.unsqueeze(0).repeat(
        history_size, *([1] * frame.ndim)
      )
      self._capture_times_s = current_time_s - torch.arange(
        history_size,
        device=frame.device,
        dtype=torch.float64,
      ) * capture_period_s
      self._next_capture_time_s = current_time_s + capture_period_s
      self._delay_s = self._sample_delay(frame.shape[0], frame.device)
      self._invalid_env_ids = None
      return self._select(current_time_s)

    if capture_due:
      assert self._frames is not None
      assert self._capture_times_s is not None
      assert self._next_capture_time_s is not None
      periods_elapsed = max(
        1,
        int(
          (current_time_s - self._next_capture_time_s + 1.0e-9)
          // capture_period_s
        )
        + 1,
      )
      capture_time_s = (
        self._next_capture_time_s + (periods_elapsed - 1) * capture_period_s
      )
      self._frames = torch.roll(self._frames, shifts=1, dims=0)
      self._frames[0].copy_(frame)
      self._capture_times_s = torch.roll(
        self._capture_times_s,
        shifts=1,
        dims=0,
      )
      self._capture_times_s[0] = capture_time_s
      self._next_capture_time_s += periods_elapsed * capture_period_s

    if reset_env_ids is not None:
      assert self._frames is not None
      assert self._delay_s is not None
      reset_env_ids = reset_env_ids.to(device=frame.device)
      reset_frame = (
        frame.index_select(0, reset_env_ids)
        if process_env_ids is None
        else frame
      )
      self._frames[:, reset_env_ids] = reset_frame.unsqueeze(0)
      self._delay_s[reset_env_ids] = self._sample_delay(
        reset_env_ids.numel(),
        frame.device,
      )
      self._invalid_env_ids = None

    return self._select(current_time_s)

  def _sample_delay(self, count: int, device: torch.device) -> torch.Tensor:
    assert self._delay_range_s is not None
    delay_min_s, delay_max_s = self._delay_range_s
    return delay_min_s + (delay_max_s - delay_min_s) * torch.rand(
      count,
      device=device,
      dtype=torch.float64,
    )

  def _select(self, current_time_s: float) -> torch.Tensor:
    assert self._frames is not None
    assert self._capture_times_s is not None
    assert self._delay_s is not None

    target_times_s = current_time_s - self._delay_s
    too_new = self._capture_times_s.unsqueeze(0) > (
      target_times_s.unsqueeze(1) + 1.0e-9
    )
    frame_indices = too_new.sum(dim=1).clamp_max(self._frames.shape[0] - 1)
    env_indices = torch.arange(self._frames.shape[1], device=self._frames.device)
    return self._frames[frame_indices, env_indices]

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._processor.reset(env_ids)
    if env_ids is None or isinstance(env_ids, slice):
      self._frames = None
      self._capture_times_s = None
      self._next_capture_time_s = None
      self._capture_period_s = None
      self._delay_s = None
      self._delay_range_s = None
      self._invalid_env_ids = None
      return
    if self._frames is None or env_ids.numel() == 0:
      return
    env_ids = env_ids.to(device=self._frames.device, dtype=torch.long)
    if self._invalid_env_ids is None:
      self._invalid_env_ids = env_ids
    else:
      self._invalid_env_ids = torch.unique(
        torch.cat((self._invalid_env_ids, env_ids))
      )


async_depth_buffer = AsyncDepthBuffer


def _resolve_grid_shape(
  num_samples: int,
  grid_shape: tuple[int, int] | None,
) -> tuple[int, int]:
  if grid_shape is not None:
    rows, cols = grid_shape
    if rows * cols != num_samples:
      raise ValueError(
        f"grid_shape={grid_shape} does not match {num_samples} height samples"
      )
    return rows, cols

  side = int(num_samples**0.5)
  if side * side != num_samples:
    raise ValueError(
      f"Cannot infer a square grid from {num_samples} height samples; "
      "pass grid_shape explicitly."
    )
  return side, side


def _terrain_clearance_samples(
  sensor: TerrainHeightSensor | RayCastSensor,
) -> torch.Tensor:
  if isinstance(sensor, TerrainHeightSensor):
    height_samples = sensor.data.heights
    if height_samples.ndim != 3:
      raise ValueError(
        "terrain_roughness_indicator requires unreduced height samples with shape "
        f"[B, F, N], got {tuple(height_samples.shape)}"
      )
    return height_samples

  if isinstance(sensor, RayCastSensor):
    data = sensor.data
    f_count = sensor.num_frames
    n_count = sensor.num_rays_per_frame
    batch_size = data.distances.shape[0]
    frame_z = data.frame_pos_w[:, :, 2:3]
    hit_z = data.hit_pos_w[..., 2].view(batch_size, f_count, n_count)
    heights = frame_z - hit_z
    miss_mask = data.distances.view(batch_size, f_count, n_count) < 0
    return torch.where(
      miss_mask,
      torch.full_like(heights, sensor.cfg.max_distance),
      heights,
    )

  raise TypeError(
    "terrain_roughness_indicator requires a TerrainHeightSensor or RayCastSensor, "
    f"got {type(sensor).__name__}"
  )


def terrain_roughness_indicator(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  wheel_radius: float = 0.127,
  gate_min: float = 0.10,
  gate_max: float = 0.40,
  grid_shape: tuple[int, int] | None = None,
) -> torch.Tensor:
  """Roughness gate from terrain clearance samples."""
  sensor = env.scene[sensor_name]
  height_samples = _terrain_clearance_samples(sensor)
  if wheel_radius <= 0.0:
    raise ValueError(f"wheel_radius ({wheel_radius}) must be positive")
  if gate_max <= gate_min:
    raise ValueError(
      f"gate_max ({gate_max}) must be greater than gate_min ({gate_min})"
    )

  height_samples = torch.nan_to_num(height_samples, nan=0.0, posinf=0.0, neginf=0.0)
  rows, cols = _resolve_grid_shape(height_samples.shape[-1], grid_shape)
  grid = height_samples.view(height_samples.shape[0], height_samples.shape[1], rows, cols)
  zeros = torch.zeros_like(height_samples[..., 0])

  if cols > 1:
    jump_x = torch.abs(grid[..., 1:] - grid[..., :-1]).amax(dim=(-1, -2))
  else:
    jump_x = zeros
  if rows > 1:
    jump_y = torch.abs(grid[..., 1:, :] - grid[..., :-1, :]).amax(dim=(-1, -2))
  else:
    jump_y = zeros
  jump = torch.maximum(jump_x, jump_y)

  if cols > 2:
    curvature_x = torch.abs(
      grid[..., :, 2:] - 2.0 * grid[..., :, 1:-1] + grid[..., :, :-2]
    ).amax(dim=(-1, -2))
  else:
    curvature_x = zeros
  if rows > 2:
    curvature_y = torch.abs(
      grid[..., 2:, :] - 2.0 * grid[..., 1:-1, :] + grid[..., :-2, :]
    ).amax(dim=(-1, -2))
  else:
    curvature_y = zeros
  curvature = torch.maximum(curvature_x, curvature_y)

  foot_roughness = torch.maximum(jump / wheel_radius, curvature / wheel_radius)
  robot_roughness = foot_roughness.max(dim=1).values
  u = torch.clamp((robot_roughness - gate_min) / (gate_max - gate_min), 0.0, 1.0)
  gate = u * u * (3.0 - 2.0 * u)
  return gate.unsqueeze(-1)


def wheel_roughness_gate(
  env: ManagerBasedRlEnv,
  sensor_name: str = "wheel_height_scan",
  wheel_radius: float = 0.127,
  gate_min: float = 0.10,
  gate_max: float = 0.40,
  grid_shape: tuple[int, int] | None = None,
) -> torch.Tensor:
  """Continuous roughness gates for left and right wheel terrain patches."""
  sensor = env.scene[sensor_name]
  height_samples = _terrain_clearance_samples(sensor)
  if height_samples.shape[1] != 2:
    raise ValueError(
      "wheel_roughness_gate requires exactly two wheel frames ordered [left, right], "
      f"got shape {tuple(height_samples.shape)}"
    )
  if wheel_radius <= 0.0:
    raise ValueError(f"wheel_radius ({wheel_radius}) must be positive")
  if gate_max <= gate_min:
    raise ValueError(
      f"gate_max ({gate_max}) must be greater than gate_min ({gate_min})"
    )

  height_samples = torch.nan_to_num(height_samples, nan=0.0, posinf=0.0, neginf=0.0)
  rows, cols = _resolve_grid_shape(height_samples.shape[-1], grid_shape)
  grid = height_samples.view(height_samples.shape[0], 2, rows, cols)
  zeros = torch.zeros_like(height_samples[..., 0])

  if cols > 1:
    jump_x = torch.abs(grid[..., 1:] - grid[..., :-1]).amax(dim=(-1, -2))
  else:
    jump_x = zeros
  if rows > 1:
    jump_y = torch.abs(grid[..., 1:, :] - grid[..., :-1, :]).amax(dim=(-1, -2))
  else:
    jump_y = zeros
  jump = torch.maximum(jump_x, jump_y)

  if cols > 2:
    curvature_x = torch.abs(
      grid[..., :, 2:] - 2.0 * grid[..., :, 1:-1] + grid[..., :, :-2]
    ).amax(dim=(-1, -2))
  else:
    curvature_x = zeros
  if rows > 2:
    curvature_y = torch.abs(
      grid[..., 2:, :] - 2.0 * grid[..., 1:-1, :] + grid[..., :-2, :]
    ).amax(dim=(-1, -2))
  else:
    curvature_y = zeros
  curvature = torch.maximum(curvature_x, curvature_y)
  roughness = torch.maximum(jump, curvature) / wheel_radius
  u = torch.clamp((roughness - gate_min) / (gate_max - gate_min), 0.0, 1.0)
  return u * u * (3.0 - 2.0 * u)


def _normalize_to_unit_range(
  value: torch.Tensor, lower: float, upper: float
) -> torch.Tensor:
  scaled = 2.0 * (value - lower) / (upper - lower) - 1.0
  return torch.clamp(scaled, -1.0, 1.0)


def domain_randomization_delta_quantity(
  env: ManagerBasedRlEnv,
  wheel_friction_event: str = "wheel_friction",
  wheel_friction_difference_event: str = "wheel_friction_difference",
  encoder_bias_event: str = "encoder_bias",
  base_com_event: str = "base_com",
) -> torch.Tensor:
  """Normalized domain-randomization quantities visible to the policy."""
  wheel_friction_cfg = env.event_manager.get_term_cfg(wheel_friction_event)
  friction_asset_cfg: SceneEntityCfg = wheel_friction_cfg.params["asset_cfg"]
  wheel_friction_common_range = wheel_friction_cfg.params["ranges"]
  wheel_friction_difference_cfg = env.event_manager.get_term_cfg(
    wheel_friction_difference_event
  )
  wheel_friction_difference_range = wheel_friction_difference_cfg.params["ranges"]
  wheel_friction_range = (
    wheel_friction_common_range[0] + wheel_friction_difference_range[0],
    wheel_friction_common_range[1] + wheel_friction_difference_range[1],
  )
  friction_asset = env.scene[friction_asset_cfg.name]
  wheel_geom_ids = friction_asset.indexing.geom_ids[friction_asset_cfg.geom_ids]
  wheel_friction = env.sim.model.geom_friction[:, wheel_geom_ids, 0]
  wheel_friction = _normalize_to_unit_range(
    wheel_friction,
    wheel_friction_range[0],
    wheel_friction_range[1],
  )

  encoder_bias_cfg = env.event_manager.get_term_cfg(encoder_bias_event)
  encoder_asset_cfg: SceneEntityCfg = encoder_bias_cfg.params["asset_cfg"]
  encoder_bias_range = encoder_bias_cfg.params["bias_range"]
  encoder_asset = env.scene[encoder_asset_cfg.name]
  encoder_bias = encoder_asset.data.encoder_bias[:, encoder_asset_cfg.joint_ids]
  encoder_bias = _normalize_to_unit_range(
    encoder_bias,
    encoder_bias_range[0],
    encoder_bias_range[1],
  )

  base_com_cfg = env.event_manager.get_term_cfg(base_com_event)
  base_com_asset_cfg: SceneEntityCfg = base_com_cfg.params["asset_cfg"]
  base_com_ranges = base_com_cfg.params["ranges"]
  base_com_asset = env.scene[base_com_asset_cfg.name]
  base_body_ids = base_com_asset.indexing.body_ids[base_com_asset_cfg.body_ids]
  current_body_ipos = env.sim.model.body_ipos[:, base_body_ids, :].reshape(
    env.num_envs, -1
  )
  default_body_ipos = env.sim.get_default_field("body_ipos")[base_body_ids, :].reshape(
    1, -1
  )
  base_com_delta = current_body_ipos - default_body_ipos
  base_com_delta = torch.stack(
    [
      _normalize_to_unit_range(base_com_delta[:, axis], *base_com_ranges[axis])
      for axis in range(3)
    ],
    dim=1,
  )

  return torch.cat((wheel_friction, encoder_bias, base_com_delta), dim=1)
