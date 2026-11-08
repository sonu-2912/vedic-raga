"""Acquire the official CC-BY-4.0 GamER release, with size/disk/checksum checks."""
import hashlib
import json
import shutil
from pathlib import Path
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
def main():
    meta = json.loads((ROOT/'reports/gamer_metadata.json').read_text())
    assert meta['metadata']['license']['id'] == 'cc-by-4.0'
    assert sum(f['size'] for f in meta['files']) < 1_000_000_000
    assert shutil.disk_usage(ROOT).free > 8_000_000_000
    folder = ROOT/'data/private/gamer'; folder.mkdir(parents=True, exist_ok=True)
    for item in sorted(meta['files'], key=lambda f:f['size']):
        dest = folder/item['key']
        if dest.exists() and dest.stat().st_size == item['size']:
            if hashlib.md5(dest.read_bytes()).hexdigest() == item['checksum'].split(':')[-1]:
                print('Already verified: '+item['key'], flush=True); continue
        if item['key']=='data.zip':
            partial=dest.with_suffix('.zip.partial')
            start=partial.stat().st_size if partial.exists() else 0
            pieces=folder/'download_parts';pieces.mkdir(exist_ok=True)
            ranges=[]
            for first in range(start,item['size'],16*1024*1024):
                end=min(first+16*1024*1024,item['size'])
                existing=pieces/f'{first:012d}.bin'
                if existing.exists() and existing.stat().st_size==end-first:ranges.append((first,end-1))
                else:ranges.extend((s,min(s+4*1024*1024,end)-1) for s in range(first,end,4*1024*1024))
            def fetch(bounds):
                first,last=bounds;target=pieces/f'{first:012d}.bin'
                if target.exists() and target.stat().st_size==last-first+1:return bounds
                for attempt in range(4):
                    try:
                        r=requests.get('https://zenodo.org/records/8191877/files/data.zip?download=1',headers={'Range':f'bytes={first}-{last}'},timeout=(30,90))
                        r.raise_for_status()
                        if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {first}-{last}/{item["size"]}' or len(r.content)!=last-first+1:raise ValueError('Range response mismatch')
                        target.write_bytes(r.content);return bounds
                    except Exception:
                        if attempt==3:raise
            with ThreadPoolExecutor(max_workers=8) as pool:
                for done in as_completed([pool.submit(fetch,b) for b in ranges]):print('Downloaded range '+str(done.result()),flush=True)
            with partial.open('ab') as stream:
                for first,last in ranges:
                    with (pieces/f'{first:012d}.bin').open('rb') as source:shutil.copyfileobj(source,stream)
            digest=hashlib.md5()
            with partial.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
            if partial.stat().st_size!=item['size'] or digest.hexdigest()!=item['checksum'].split(':')[-1]:raise ValueError('Download integrity failure')
            partial.replace(dest);print('Verified: '+item['key'],flush=True);continue
        partial = dest.with_suffix(dest.suffix+'.partial'); digest = hashlib.md5(); count=0; next_report=50_000_000
        with requests.get(item['links']['self'], stream=True, timeout=(30,120)) as response:
            response.raise_for_status()
            with partial.open('wb') as stream:
                for block in response.iter_content(1024*1024):
                    stream.write(block);digest.update(block);count+=len(block)
                    if count>=next_report:
                        print(f'{item["key"]}: {count}/{item["size"]} bytes',flush=True);next_report+=50_000_000
        if count!=item['size'] or digest.hexdigest()!=item['checksum'].split(':')[-1]:raise ValueError('Download integrity failure')
        partial.replace(dest);print('Verified: '+item['key'],flush=True)
if __name__=='__main__':main()
