"""Preserve each atomic epoch state for validation-only selection/audit."""
import argparse,hashlib,json,time
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args()
    run=Path(a.run_dir);dest=run/'epoch_snapshots';dest.mkdir(exist_ok=True)
    seen=set();index=[];start=time.monotonic()
    while time.monotonic()-start<4*3600:
        source=run/'latest.pt'
        if source.exists():
            data=source.read_bytes();digest=hashlib.sha256(data).hexdigest()
            if digest not in seen:
                target=dest/f'{len(index):03d}-{digest[:12]}.pt'
                target.write_bytes(data);seen.add(digest);index.append({'file':str(target),'sha256':digest})
                (dest/'index.json').write_text(json.dumps(index,indent=2));print('Preserved '+str(target),flush=True)
        if (run/'completion.json').exists():break
        time.sleep(10)
if __name__=='__main__':main()
