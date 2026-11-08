"""Temporary Windows idle-sleep request during a declared training run."""
import argparse,ctypes,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--minutes',type=float,default=125);a=p.parse_args();root=Path(__file__).resolve().parents[1];run=(root/a.run_dir).resolve()
if not run.is_relative_to((root/'reports').resolve()):raise ValueError('Expected a project reports directory')
kernel=ctypes.WinDLL('kernel32',use_last_error=True);set_state=kernel.SetThreadExecutionState;set_state.argtypes=[ctypes.c_uint];set_state.restype=ctypes.c_uint
try:
 if not set_state(0x80000001):raise OSError(ctypes.get_last_error(),'Temporary awake request failed')
 deadline=time.monotonic()+a.minutes*60
 while not (run/'completion.json').exists() and time.monotonic()<deadline:time.sleep(10)
finally:set_state(0x80000000)
