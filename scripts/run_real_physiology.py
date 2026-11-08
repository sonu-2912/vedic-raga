"""Wait for the checksum-verified release, then execute both alignment analyses."""
import subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
deadline=time.monotonic()+1800
while not (root/'data/private/gamer/data.zip').exists():
    if time.monotonic()>deadline:raise TimeoutError('Official dataset download did not complete')
    time.sleep(5)
for args in [[],['--sensitivity']]:
    subprocess.run([sys.executable,'-u','-m','scripts.gamer_experiment',*args],cwd=root,check=True)
