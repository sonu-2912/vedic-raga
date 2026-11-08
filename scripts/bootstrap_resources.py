"""Provision the separately attributed catalog; no upstream assets are republished."""
import argparse,hashlib,json,shutil,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXPECTED='2675283a6bd6d5a415150cf023a782424342c8c051247afd68fa622b2b5de05c'
URL='https://raw.githubusercontent.com/alronova/raag-ratnakar/main/raga_sthayi_bhava_dataset_enriched.json'
def main():
 p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True);g.add_argument('--source-dir',type=Path);g.add_argument('--download',action='store_true');a=p.parse_args()
 target=ROOT/'external/raag-ratnakar-main/raga_sthayi_bhava_dataset_enriched.json'
 if target.exists():data=target.read_bytes()
 elif a.source_dir:data=(a.source_dir/target.name).read_bytes()
 else:
  with urllib.request.urlopen(URL,timeout=60) as response:data=response.read(5*1024*1024+1)
  if len(data)>5*1024*1024:raise SystemExit('Unexpected catalog size; original file preserved.')
 if hashlib.sha256(data).hexdigest()!=EXPECTED:raise SystemExit('Catalog differs from inspected source. Review the upstream change before updating the pinned checksum.')
 rows=json.loads(data);assert len(rows)==72 and sum(len(v) for r in rows for v in r['youtube_links'].values())==528
 target.parent.mkdir(parents=True,exist_ok=True)
 if not target.exists():target.write_bytes(data)
 print('Verified local catalog:72 ragas,528 link entries. Upstream licence unresolved; see THIRD_PARTY_NOTICES.md. Optional Navrasa weights are provisioned separately.')
if __name__=='__main__':main()
