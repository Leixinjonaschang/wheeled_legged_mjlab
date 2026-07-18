# WF-TRON1B Depth Sim2Real 域随机化实现计划

针对任务 `Mjlab-Velocity-Rough-WF-Tron1B-RepTS-LinVel-Depth`（训练仓库 `wheeled_legged_mjlab` @ `develop`）。

## 锁定的决策

- capture 频率：25 → **30 Hz**（实机确认，和 LIPM paper 同款 D435i）。
- 相机 DR：**`reset` 每回合重采样**；相机位置**三轴各自独立 ±10 mm**；pitch ±1°；FOV ±1°。
- depth system delay：**异步时钟叠加**，`[0, 20] ms`，per-env、`reset` 重采样。

## 数据来源（论文 arXiv 2509.09106v2, Table III）

| 参数 | 范围 | 单位 | MuJoCo/mjlab 实际值 |
|---|---|---|---|
| Camera position | [−10, 10] | mm | ±0.010 **m**（三轴独立） |
| Camera pitch | [−1, 1] | deg | ±0.017453 **rad** |
| Camera FOV | [−1, 1] | deg | ±1.0 **deg**（cam_fovy 就是度） |
| System delay | [0, 20] | ms | 0.0–0.020 **s** |

## 改动文件总览

1. `src/wheeled_legged_mjlab/tasks/velocity/config/wf_tron1b/env_cfgs.py`
   - 常量 `DEPTH_CAPTURE_FREQUENCY_HZ` 30.0；`DEPTH_SYSTEM_DELAY_RANGE_S` 新增。
   - `make_events()` 加 `depth` 门控 + 3 个相机 DR 事件。
   - async depth 观测项传入 `system_delay_range_s`。
   - `apply_play_overrides()` 关掉相机 DR 与 delay。
2. `src/wheeled_legged_mjlab/tasks/velocity/mdp/observations.py`
   - `AsyncDepthBuffer` 扩展：环形历史 + per-env 延迟 + 按 `now−delay` 取帧。
3. 测试：`tests/test_representation_teacher_student_config.py` + `observations` 的单测。
4. （可选）deploy 仓库 sim2sim `MJLAB_DEPTH_CAPTURE_HZ=30` 做 parity。

---

## 1. capture 频率 → 30 Hz

`env_cfgs.py` 常量区（现 `:86`）：

```python
DEPTH_CAPTURE_FREQUENCY_HZ = 30.0        # was 25.0
DEPTH_SYSTEM_DELAY_RANGE_S = (0.0, 0.020)  # LIPM Table III system delay
```

async depth 观测组（现 `:396-408`）把 delay range 传进去：

```python
if depth and async_depth:
    observations[DEPTH_CAMERA_NAME] = ObservationGroupCfg(
        terms={
            DEPTH_CAMERA_NAME: ObservationTermCfg(
                func=mdp.async_depth_buffer,
                params={
                    "sensor_name": DEPTH_CAMERA_NAME,
                    "capture_frequency_hz": DEPTH_CAPTURE_FREQUENCY_HZ,
                    "system_delay_range_s": DEPTH_SYSTEM_DELAY_RANGE_S,
                },
            )
        },
        concatenate_terms=True,
        enable_corruption=False,
    )
```

观测形状不变，仍是单帧 `[N,1,28,48]`，模型/ONNX/deploy 接口都不用动。

---

## 2. 相机参数 DR（3 个 event，`reset` 模式）

### 2.1 `make_events()` 加 depth 门控

签名改为 `def make_events(*, depth: bool = False)`，调用处（现 `:943`）改成 `events=make_events(depth=depth)`。在返回的 dict 末尾追加：

```python
if depth:
    events["cam_pos"] = EventTermCfg(
        func=mdp.dr.cam_pos,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(ROBOT_ENTITY),  # 机器人实体上只有 d435
            "operation": "add",                          # 必须 add，见 2.3
            "ranges": {0: (-0.010, 0.010),               # 三轴各自独立 ±10mm（米）
                       1: (-0.010, 0.010),
                       2: (-0.010, 0.010)},
        },
    )
    events["cam_pitch"] = EventTermCfg(
        func=mdp.dr.cam_quat,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(ROBOT_ENTITY),
            "pitch_range": (-0.0174533, 0.0174533),      # ±1°，弧度
            "roll_range": (0.0, 0.0),
            "yaw_range": (0.0, 0.0),
        },
    )
    events["cam_fovy"] = EventTermCfg(
        func=mdp.dr.cam_fovy,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(ROBOT_ENTITY),
            "operation": "add",                          # 必须 add
            "ranges": (-1.0, 1.0),                       # ±1°，度
        },
    )
```

这三个函数是 mjlab 1.3.0 自带的 per-env DR（`mjlab/envs/mdp/dr/camera.py`，已在 `mdp.dr` 导出），和现有 `mdp.dr.body_com_offset` 同一套机制，改的是渲染前的 per-env mjModel 相机字段。

### 2.2 单位换算（最易错）

- 位置：MuJoCo 是**米**，±10 mm = **±0.010**。
- pitch：`cam_quat` 的 range 是**弧度**，±1° = **±0.0174533**。
- FOV：`cam_fovy` 是**度**，直接 **±1.0**。

### 2.3 三个必须注意的坑

1. **`operation="add"` 必须显式写。** `cam_pos`/`cam_fovy` 默认 `operation="abs"`，会把 fovy 直接设成 [−1,1] 度（≈0，画面全毁）、把相机位置设成绝对 [−0.01,0.01] m。`cam_quat` 无 operation 参数——它内部相对 default quat 组合扰动，安全。
2. **pitch 到底哪根轴，实测确认。** `cam_quat` 用 `quat_from_euler_xyz` 把 (roll,pitch,yaw) 组合到 default quat；而 d435 安装 euler 是 `0 -0.507 -1.5708`（`robot.xml:49`，非平凡）。落地前只开 pitch、跨 env 渲染对比，确认动的是光轴上下俯仰；若发现是 roll 在动，把范围挪到 `roll_range`。
3. **asset_cfg 作用域。** 确认 `SceneEntityCfg(ROBOT_ENTITY)` 解析到的相机只有 d435（worldbody 的 `track` 相机不属于机器人实体）。若 mjlab 报缺 model field，说明该字段没按 per-env 暴露——`@requires_model_fields` 装饰器一般会自动登记，报错再排查。

---

## 3. 异步时钟叠加延迟（扩展 `AsyncDepthBuffer`）

### 3.1 原理

延迟挂在**相机 30 Hz 捕获时钟**上、以连续毫秒表示（不被 20 ms 的 policy 步量化）。保留最近若干**带全局时间戳**的捕获帧；查询时对每个 env 返回"捕获时间 ≤ now − delay_i 的最新那帧"。

- 捕获是全局的（sim 里同一相机时钟），所以时间戳是标量、只有 `delay_i` per-env。
- 历史槽数 `n_hist = ceil(max_delay / capture_period) + 1`。20 ms / 33.3 ms → **2 槽**够用；用公式写死可自动适配未来更大延迟或更高帧率。
- 效果：一个捕获周期内，每个 env 在 `t_capture + delay_i` 时刻从"上一帧"切到"新帧"，即对帧到达施加 delay_i 的 latency，且自然叠在已有的 ZOH 保持之上。

### 3.2 替换后的类（`observations.py`，2 空格缩进对齐现有风格）

```python
class AsyncDepthBuffer:
  """Latest depth frame on an independent capture clock, with optional per-env system delay."""

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    del cfg, env
    self._frames: torch.Tensor | None = None        # [n_hist, N, 1, H, W] newest-first
    self._capture_times_s: torch.Tensor | None = None  # [n_hist] global capture time, newest-first
    self._next_capture_time_s: float | None = None
    self._delay_s: torch.Tensor | None = None        # [N] per-env delay (seconds)
    self._delay_range_s: tuple[float, float] = (0.0, 0.0)
    self._n_hist: int = 1
    self._invalid_env_ids: torch.Tensor | None = None

  def _sample_delay(self, n: int, device) -> torch.Tensor:
    lo, hi = self._delay_range_s
    return lo + (hi - lo) * torch.rand(n, device=device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str = "depth_camera",
    capture_frequency_hz: float = 30.0,
    system_delay_range_s: tuple[float, float] = (0.0, 0.0),
  ) -> torch.Tensor:
    if capture_frequency_hz <= 0.0:
      raise ValueError(f"capture_frequency_hz must be positive, got {capture_frequency_hz}")
    step_dt = float(env.step_dt)
    capture_period_s = 1.0 / capture_frequency_hz
    now = int(getattr(env, "common_step_counter", 0)) * step_dt
    self._delay_range_s = (float(system_delay_range_s[0]), float(system_delay_range_s[1]))
    n_hist = max(1, math.ceil(self._delay_range_s[1] / capture_period_s) + 1)

    needs_init = self._frames is None or self._n_hist != n_hist
    capture_due = self._next_capture_time_s is None or now + 1e-9 >= self._next_capture_time_s
    needs_reset_fill = self._invalid_env_ids is not None

    if not (needs_init or capture_due or needs_reset_fill):
      return self._select(now)

    frame = depth_image(env, sensor_name=sensor_name).unsqueeze(1)  # [N,1,H,W]
    N, dev = frame.shape[0], frame.device

    if needs_init:
      self._n_hist = n_hist
      self._frames = frame.unsqueeze(0).repeat(n_hist, *([1] * frame.ndim))   # [n_hist,N,1,H,W]
      self._capture_times_s = torch.full((n_hist,), now, device=dev)
      self._next_capture_time_s = now + capture_period_s
      self._delay_s = self._sample_delay(N, dev)
      self._invalid_env_ids = None
      return self._select(now)

    if capture_due:
      self._frames = torch.roll(self._frames, shifts=1, dims=0)   # 老帧后移，slot0 放新帧
      self._frames[0] = frame
      self._capture_times_s = torch.roll(self._capture_times_s, shifts=1, dims=0)
      self._capture_times_s[0] = now
      periods = max(1, int((now - self._next_capture_time_s + 1e-9) // capture_period_s) + 1)
      self._next_capture_time_s += periods * capture_period_s
      self._invalid_env_ids = None

    if needs_reset_fill:
      env_ids = self._invalid_env_ids.to(device=dev, dtype=torch.long)
      self._frames[:, env_ids] = frame[env_ids].unsqueeze(0)      # 所有历史槽填当前帧
      self._delay_s[env_ids] = self._sample_delay(env_ids.numel(), dev)  # 重采样延迟
      self._invalid_env_ids = None

    return self._select(now)

  def _select(self, now: float) -> torch.Tensor:
    assert self._frames is not None and self._capture_times_s is not None and self._delay_s is not None
    ages = now - self._capture_times_s                             # [n_hist]
    too_fresh = ages.unsqueeze(0) < self._delay_s.unsqueeze(1)     # [N, n_hist]
    k = too_fresh.sum(dim=1).clamp_max(self._n_hist - 1)           # [N] 最新的"够老"槽
    env_idx = torch.arange(self._frames.shape[1], device=self._frames.device)
    return self._frames[k, env_idx]                               # [N,1,H,W]

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None or isinstance(env_ids, slice):
      self._frames = None
      self._capture_times_s = None
      self._next_capture_time_s = None
      self._delay_s = None
      self._invalid_env_ids = None
      return
    if self._frames is None or env_ids.numel() == 0:
      return
    env_ids = env_ids.to(device=self._frames.device, dtype=torch.long)
    self._invalid_env_ids = (
      env_ids if self._invalid_env_ids is None
      else torch.unique(torch.cat((self._invalid_env_ids, env_ids)))
    )
```

顶部需要 `import math`（若未引入）。`async_depth_buffer = AsyncDepthBuffer` 保持不变。

### 3.3 关键点

- **默认 `system_delay_range_s=(0.0, 0.0)`** → `n_hist=1`、delay 恒 0，行为与现状完全一致（向后兼容，老测试不破）。
- **延迟重采样复用现有 `_invalid_env_ids` 通路**：`reset(env_ids)` 只标记失效，真正的填帧 + 重采样在 `__call__` 里做（因为要用到当帧渲染和 device/N），和现有帧填充逻辑一致。
- 捕获全局、延迟 per-env；`_select` 用"数有多少槽太新"来定位每个 env 该取的槽，向量化、无循环。

---

## 4. play override

`apply_play_overrides()`（现 `:960+`）里，评估时关掉相机 DR 与 delay 以便复现：

```python
for name in ("cam_pos", "cam_pitch", "cam_fovy"):
    cfg.events.pop(name, None)
if DEPTH_CAMERA_NAME in cfg.observations:
    term = cfg.observations[DEPTH_CAMERA_NAME].terms[DEPTH_CAMERA_NAME]
    term.params["system_delay_range_s"] = (0.0, 0.0)
```

---

## 5. 测试

### 5.1 渲染冒烟（最高风险，先做）
2 个 env 给不同 `cam_fovy`/`cam_pos`，断言渲染出的 depth 不同 → 验证 mjlab Warp 渲染器确实按 per-env 相机字段渲染。这是整套相机 DR 成立的前提，务必先单独确认。

### 5.2 配置测试（`tests/test_representation_teacher_student_config.py` 风格）
- depth 任务注册了 `cam_pos/cam_pitch/cam_fovy`，`mode=="reset"`，range/单位正确（pos 0.010、pitch 0.0174533、fovy 1.0）。
- 非 depth 任务不含这三个事件。
- async 观测项 `capture_frequency_hz==30.0`、`system_delay_range_s==(0.0,0.020)`。

### 5.3 async 延迟单测（用 mock env：`common_step_counter`、`step_dt`、桩 `depth_image`）
- `range=(0,0)`：输出恒等于最新捕获帧（复现现状）。
- `range>0`：同一捕获周期内，`now−t_capture < delay` 时返回上一帧，`≥ delay` 后返回新帧。
- `reset(env_ids)`：该 env 延迟被重采样、历史槽填当前帧；其它 env 不受影响。
- 形状恒为 `[N,1,H,W]`。

---

## 6. deploy 一致性

- 相机 DR 是**训练专属**，deploy（`wf_tron1b_deploy`）**不需要改**。
- 延迟：deploy 已按最新 ROS 帧 + `max_age_s=0.5` 消费，训练的 0–20 ms 远在其内，**不必改**；要严格 parity 才在 deploy 侧加同等 hold。
- sim2sim：可把 deploy 仓库 `MJLAB_DEPTH_CAPTURE_HZ` 设 30 与训练捕获率对齐（可选）。

---

## 7. 实施顺序

1. 常量 30 Hz + `DEPTH_SYSTEM_DELAY_RANGE_S`（改动最小，先跑通训练不报错）。
2. 渲染冒烟测试（5.1）——先确认 per-env 相机渲染成立，再往下。
3. `make_events` 加 3 个相机 DR + depth 门控（5.2 配置测试）。
4. `AsyncDepthBuffer` 扩展 + 延迟单测（5.3）。
5. play override。
6. 短训练 sanity：看 `Loss/lin_vel`、RMSE 是否稳定，depth 分支未退化。

## 待实测确认（不阻塞开工）

- pitch 对应的 RPY 轴（2.3 第 2 点）。
- mjlab 渲染器 per-env 相机字段（5.1）。
