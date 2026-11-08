"""Temporary Windows auto-sleep request during this bounded CPU stage; no persistent settings."""
import ctypes,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
kernel=ctypes.WinDLL('kernel32',use_last_error=True);set_state=kernel.SetThreadExecutionState;set_state.argtypes=[ctypes.c_uint];set_state.restype=ctypes.c_uint
try:
 if not set_state(0x80000001):raise OSError(ctypes.get_last_error(),'Cannot request temporary awake state')
 deadline=time.monotonic()+90*60
 while not (ROOT/'reports/pure_native_residual_20261108/completion.json').exists() and time.monotonic()<deadline:time.sleep(10)
finally:set_state(0x80000000)
