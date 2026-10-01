#!/usr/bin/env bash
# mode_switch_v2 batch: 10 rough types x 3 checkpoints = 30 rollouts. Each rollout: 1024 envs =
# 4 difficulty levels (0.25/0.5/0.75/1.0 x the training maximum) x 16 lanes x 16 robots,
# forward_wide commands (heading 0, v_x ~ U(0.5, 2.0) m/s), 40 s time limit (2000 policy steps),
# eval seed 20260928. Then aggregate (summary.json, trajectories.csv, identity_checks.json,
# levels.json, stats.json) and the per-env checks (terrain fingerprint = CPU audit, spawn row/lane,
# commands, lateral drift).
# Noise floor: rollouts are not bit-reproducible (GPU physics), so Ours is run a second time with the
# identical configuration into replicate/ (REPLICATE=0 skips it) and tools/noise_floor.py writes
# noise_floor.json: run-to-run differences per cell, to be read next to the stats.json intervals.
#
# Resumable: finished rollouts (<name>.npz; written as .npz.part and renamed on success) are
# skipped, so a failed or interrupted batch is completed by running this script again.
# Concurrency: JOBS=3 (override: JOBS=2 ./run_all.sh). Measured on the L40S (46 GB): one 1024-env
# rollout peaks at ~4.1 GB, 3 concurrent at 11.4 GB (25 %), 8 concurrent at 26.4 GB (57 %), so
# memory keeps >= 20 % headroom for any JOBS <= 8; but GPU utilisation is ~100 % from 2-3 jobs on
# (aggregate throughput flat beyond that), so more jobs only add CPU/GPU contention.
# Expected wall time: ~35 min, ~47 min with the replicate (3 concurrent full-length rollouts took 203 s).
set -u -o pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
OUT=$REPO/logs/lipm_eval/mode_switch_v2
LOGS=$OUT/logs
JOBS=${JOBS:-3}
REPLICATE=${REPLICATE:-1}
COMMAND=forward_wide
ROUGHS=(tilted_grid random_spread discrete_obstacles random_rough hf_pyramid_slope
        hf_pyramid_slope_inv pyramid_stair pyramid_stair_inv random_stairs stepping_stones)
CKPTS=(Ours RGGP LPGP)

cd "$REPO" || { echo "cannot cd to $REPO" >&2; exit 1; }
mkdir -p "$LOGS"

run_one() {  # rough ckpt [subdirectory of OUT, e.g. replicate]
  local rough=$1 ckpt=$2 sub=${3:-}
  local name=${rough}__${COMMAND}__${ckpt}
  local dest=$OUT${sub:+/$sub} log=$LOGS/${sub:+${sub}__}$name.log
  if [[ -f $dest/$name.npz ]]; then
    echo "[skip]  ${sub:+$sub/}$name (npz exists)"
    return 0
  fi
  echo "[start] ${sub:+$sub/}$name $(date +%T)"
  .venv/bin/python scripts/eval/mode_switch_eval.py rollout --ckpt "$ckpt" --rough "$rough" \
    --command "$COMMAND" --levels 4 --num-envs 1024 --time-limit 40 --out "$dest" \
    > "$log" 2>&1
  local status=$?
  if [[ $status -eq 0 && -f $dest/$name.npz ]]; then
    echo "[done]  ${sub:+$sub/}$name $(date +%T) ($(grep -h '^recorded' "$log"))"
    return 0
  fi
  echo "[FAIL]  ${sub:+$sub/}$name exit $status, log: $log" >&2
  tail -n 5 "$log" | sed 's/^/        /' >&2
  return 1
}
export -f run_one
export OUT LOGS COMMAND

echo "mode_switch_v2: ${#ROUGHS[@]} rough types x ${#CKPTS[@]} ckpts, $JOBS concurrent, start $(date)"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
{
  for rough in "${ROUGHS[@]}"; do
    for ckpt in "${CKPTS[@]}"; do
      printf '%s %s -\n' "$rough" "$ckpt"
    done
  done
  if (( REPLICATE )); then
    for rough in "${ROUGHS[@]}"; do printf '%s Ours replicate\n' "$rough"; done
  fi
} | xargs -P "$JOBS" -n 3 bash -c 'run_one "$1" "$2" "${3#-}"' _

# Judge completeness by the files, not by xargs (also catches killed workers).
missing=()
expected=$(( ${#ROUGHS[@]} * ${#CKPTS[@]} ))
for rough in "${ROUGHS[@]}"; do
  for ckpt in "${CKPTS[@]}"; do
    [[ -f $OUT/${rough}__${COMMAND}__${ckpt}.npz ]] || missing+=("${rough}__${COMMAND}__${ckpt}")
  done
  if (( REPLICATE )); then
    [[ -f $OUT/replicate/${rough}__${COMMAND}__Ours.npz ]] || missing+=("replicate/${rough}__${COMMAND}__Ours")
  fi
done
(( REPLICATE )) && expected=$(( expected + ${#ROUGHS[@]} ))
if (( ${#missing[@]} )); then
  echo "FAILED: ${#missing[@]} of $expected rollouts missing (rerun this script to resume):" >&2
  for name in "${missing[@]}"; do echo "  $name -> $LOGS/${name/\//__}.log" >&2; done
  exit 1
fi
echo "all $expected rollouts present, aggregating $(date +%T)"

.venv/bin/python scripts/eval/mode_switch_eval.py aggregate --out "$OUT" > "$LOGS/aggregate.log" 2>&1
agg=$?
tail -n 12 "$LOGS/aggregate.log"
if (( agg )); then
  echo "FAILED: aggregate exit $agg (identity mismatch or error), see $LOGS/aggregate.log" >&2
  exit 1
fi
: > "$LOGS/per_env_checks.log"
checks=("$OUT")
(( REPLICATE )) && checks+=("$OUT/replicate")
chk=0
for dir in "${checks[@]}"; do
  .venv/bin/python scripts/eval/mode_switch/tools/per_env_checks.py "$dir" >> "$LOGS/per_env_checks.log" 2>&1 || chk=1
done
if (( chk )); then
  echo "FAILED: per-env checks, see $LOGS/per_env_checks.log" >&2
  grep -B1 FAIL "$LOGS/per_env_checks.log" >&2
  exit 1
fi
grep WARN "$LOGS/per_env_checks.log" || true
if (( REPLICATE )); then
  if ! .venv/bin/python scripts/eval/mode_switch/tools/noise_floor.py "$OUT" "$OUT/replicate" > "$LOGS/noise_floor.log" 2>&1; then
    echo "FAILED: noise floor (replicate configuration differs?), see $LOGS/noise_floor.log" >&2
    exit 1
  fi
  head -n 7 "$LOGS/noise_floor.log"
fi
outputs="summary.json, stats.json, trajectories.csv, identity_checks.json, levels.json"
(( REPLICATE )) && outputs+=", noise_floor.json"
echo "per-env checks passed; outputs in $OUT ($outputs); done $(date)"
