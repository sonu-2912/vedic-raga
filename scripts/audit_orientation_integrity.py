"""Exact mirrored-view split-integrity audit; no held-out pixels or labels used."""
import argparse,hashlib,json
from pathlib import Path
from scripts.colab_common import fixed_split,digest,write_json
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--verify-existing',action='store_true');a=p.parse_args();run=ROOT/a.run_dir;m=json.loads((ROOT/'reports/fer_training_manifest.json').read_text(encoding='utf-8'));ds=fixed_split(ROOT/'data/private/fer2013-v1.zip',m,'train');val={r['pixel_sha256'] for r in m['val']};test={r['pixel_sha256'] for r in json.loads((run/'baseline_test_manifest.json').read_text(encoding='utf-8'))['records']};train={r['pixel_sha256'] for r in m['train']};counts={'validation':0,'historical_test_input_hashes':0,'train':0}
 for i in range(len(ds)):
  image,_=ds.raw_image(i);h=hashlib.sha256(image[:,::-1].tobytes()).hexdigest()
  for name,pool in [('validation',val),('historical_test_input_hashes',test),('train',train)]:counts[name]+=int(h in pool)
 result={'n_train':len(ds),'manifest_sha256':digest(ROOT/'reports/fer_training_manifest.json'),'overlap_counts':counts,'test_pixels_opened':False,'test_labels_used_to_select_training':False,'policy':'Reject mirror augmentation if exact held-out input overlaps exist. This is not an identity/near-duplicate audit.'}
 if a.verify_existing:assert result['overlap_counts']==json.loads((run/'mirror_integrity_audit.json').read_text(encoding='utf-8'))['overlap_counts']
 out=run/('mirror_integrity_recheck.json' if a.verify_existing else 'mirror_integrity_generated.json')
 if out.exists():raise RuntimeError('Preserve previous audit; use a new run directory')
 write_json(out,result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
