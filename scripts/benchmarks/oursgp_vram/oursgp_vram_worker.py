import atexit
import json
import runpy
import sys
import torch

@atexit.register
def report():
    if torch.cuda.is_initialized():
        print('TORCH_MEMORY ' + json.dumps({
            'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
            'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
            'final_allocated_mib': torch.cuda.memory_allocated() / 2**20,
            'final_reserved_mib': torch.cuda.memory_reserved() / 2**20,
        }), flush=True)
sys.argv = ['scripts/rsl_rl/train.py', *sys.argv[1:]]
runpy.run_path(sys.argv[0], run_name='__main__')
