"""Audit actual recorded results and live APIs without retraining or test tuning."""
import hashlib,json
from pathlib import Path
import numpy as np
import requests
root=Path(__file__).resolve().parents[1]
def read(name):return json.loads((root/name).read_text())
def verify_counts(report):
    y=np.array(report['predictions']['truth']);groups=np.array(report['predictions']['subjects']);labels=report['labels']
    assert len(y)==report['n_intervals'] and len(set(groups))==report['n_subjects']
    for method,m in report['results'].items():
        prediction=np.array(report['predictions'][method]);matrix=np.array([[np.sum((y==a)&(prediction==b)) for b in labels] for a in labels])
        assert matrix.tolist()==m['confusion_matrix']
        assert abs(float(np.trace(matrix)/matrix.sum())-m['accuracy'])<1e-12
        precision=np.divide(np.diag(matrix),matrix.sum(0),out=np.zeros(len(labels)),where=matrix.sum(0)>0)
        recall=np.divide(np.diag(matrix),matrix.sum(1),out=np.zeros(len(labels)),where=matrix.sum(1)>0)
        f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(len(labels)),where=(precision+recall)>0)
        assert abs(float(f1.mean())-m['macro_f1'])<1e-12
    assert len(report['folds'])==len(set(groups))
    assert all(f['held_out_subject'] not in f['train_subjects'] for f in report['folds'])
    ids=[i for f in report['folds'] for i in f['test_annotation_ids']]
    assert len(ids)==len(set(ids))==len(y)

def main():
    for suffix in ['', '_plus_offset']:
        features=read(f'reports/gamer_features{suffix}.json');assert len(features['records'])==len({r['annotation_id'] for r in features['records']})
        for report in read(f'reports/gamer_evaluation{suffix}.json')['tasks'].values():verify_counts(report)
    metadata=read('reports/gamer_metadata.json');checksums={}
    for f in metadata['files']:
        path=root/'data/private/gamer'/f['key'];h=hashlib.md5()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        assert path.stat().st_size==f['size'] and 'md5:'+h.hexdigest()==f['checksum'];checksums[f['key']]=f['checksum']
    base='http://127.0.0.1:8000';health=requests.get(base+'/api/health',timeout=15);health.raise_for_status()
    face=requests.get(base+'/api/research/results',timeout=30);face.raise_for_status();face=face.json()
    sha=hashlib.sha256((root/'models/fusdense.pt').read_bytes()).hexdigest();assert face['checkpoint']['sha256']==face['evaluation']['checkpoint_sha256']==sha
    phys=requests.get(base+'/api/research/physiology',timeout=15);phys.raise_for_status();phys=phys.json();assert phys['status']=='evaluated' and not phys['synthetic']
    assert all('predictions' not in t and 'folds' not in t for t in phys['tasks'].values())
    with (root/'data/demo_face.png').open('rb') as f:r=requests.post(base+'/api/emotion/predict',files={'image':('public_demo.png',f,'image/png')},data={'model':'fusdense'},timeout=60)
    r.raise_for_status();prediction=r.json();assert len(prediction['probabilities'])==7 and prediction['held_out_evaluation']['checkpoint_sha256']==sha
    (root/'reports/active_model_api_check.json').write_text(json.dumps({'input':'Existing public NASA/scikit-image demo fixture; not the user photograph','response':prediction},indent=2))
    result={'dataset_checksums':checksums,'paired_reports_audited':2,'subject_overlap':False,'duplicate_accepted_annotation_ids':False,'confusion_and_macro_f1_independently_recomputed':True,'live_api_health':health.status_code,'live_model_sha256':sha,'live_physiology_status':phys['status'],'live_face_prediction_status':r.status_code,'private_user_images_used':False}
    (root/'reports/real_results_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
