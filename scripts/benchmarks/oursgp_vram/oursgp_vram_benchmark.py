import csv
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import threading
import time

root = Path.cwd()
out = root / 'logs/benchmarks/oursgp_vram_20260926'
task = 'Mjlab-Velocity-Rough-WF-Tron1B-RepTS-LinVel-Depth-Predict-OursGP'
results = []
for count in (1024, 2048, 4096):
    cmd = [str(root / '.venv/bin/python'), '-u', str(Path(__file__).resolve().parent / 'oursgp_vram_worker.py'), task,
           '--env.scene.num-envs', str(count), '--agent.max-iterations', '10',
           '--agent.logger', 'tensorboard', '--agent.run-name', f'vram_bench_{count}',
           '--agent.experiment-name', 'oursgp_vram_benchmark']
    env = dict(os.environ, PYTHONUNBUFFERED='1', MPLCONFIGDIR='/tmp/oursgp_mpl')
    start = time.monotonic()
    state = {'iteration': -1}
    samples = []
    torch_stats = {}
    print(f'START {count}: {cmd}', flush=True)
    with (out / f'{count}_train.log').open('w') as log:
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        def consume():
            for line in proc.stdout:
                log.write(line)
                log.flush()
                match = re.search(r'Learning iteration\s+(\d+)', line)
                if match:
                    state['iteration'] = int(match.group(1))
                    print(f'{count}: completed iteration {state["iteration"]}, elapsed={time.monotonic()-start:.1f}s', flush=True)
                if line.startswith('TORCH_MEMORY '):
                    torch_stats.update(json.loads(line[len('TORCH_MEMORY '):]))
        reader = threading.Thread(target=consume)
        reader.start()
        with (out / f'{count}_gpu.csv').open('w') as f:
            writer = csv.writer(f)
            writer.writerow(['elapsed_s', 'last_completed_iteration', 'used_mib', 'total_mib', 'gpu_util_percent'])
            while proc.poll() is None:
                query = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits', '-i', '0'], capture_output=True, text=True)
                if query.returncode == 0:
                    used, total, util = map(int, query.stdout.strip().split(','))
                    row = [round(time.monotonic()-start, 3), state['iteration'], used, total, util]
                    samples.append(row)
                    writer.writerow(row)
                    f.flush()
                time.sleep(0.2)
        reader.join()
    steady = [s[2] for s in samples if 2 <= s[1] < 9]
    result = dict(num_envs=count, returncode=proc.returncode, elapsed_s=time.monotonic()-start,
                  last_completed_iteration=state['iteration'], sample_interval_s=0.2,
                  gpu_peak_mib=max((s[2] for s in samples), default=None),
                  steady_min_mib=min(steady, default=None), steady_max_mib=max(steady, default=None),
                  steady_median_mib=statistics.median(steady) if steady else None,
                  torch_memory=torch_stats, command=cmd)
    results.append(result)
    (out / 'results.json').write_text(json.dumps(results, indent=2))
    print('RESULT ' + json.dumps(result), flush=True)
