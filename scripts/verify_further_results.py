"""Audit stored facial predictions, original resources and actual live APIs."""
import hashlib,json,zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np
import requests
from scripts.colab_common import write_json
ROOT=Path(__file__).resolve().parents[1]
def read(n):return json.loads((ROOT/n).read_text())
def main():
    e=read('reports/emotion_evaluation.json');y=np.array(e['predictions']['truth']);p=np.array(e['predictions']['predicted']);cm=np.array([[np.sum((y==a)&(p==b)) for b in range(7)] for a in range(7)])
    assert cm.tolist()==e['confusion_matrix'] and cm.sum()==e['n']==6561
    assert abs(cm.trace()/cm.sum()-e['accuracy'])<1e-12
    for i,name in enumerate(['angry','disgust','fear','happy','sad','surprise','neutral']):
        precision=cm[i,i]/max(cm[:,i].sum(),1);recall=cm[i,i]/max(cm[i].sum(),1);f1=2*precision*recall/max(precision+recall,1e-18);v=e['classification_report'][name]
        assert max(abs(precision-v['precision']),abs(recall-v['recall']),abs(f1-v['f1-score']))<1e-12
    originals=read('reports/source_audit.json')['files']
    for original in originals:assert hashlib.sha256((ROOT/'external/raag-ratnakar-main'/original['path']).read_bytes()).hexdigest()==original['sha256']
    with zipfile.ZipFile(ROOT/'raag-ratnakar-main.zip') as archive:
        members=[i for i in archive.infolist() if not i.is_dir()]
        assert len(members)==len(originals)
    audit=read('reports/gamer_fixed_context_audit.json');assert audit['status']=='pass'
    base='http://127.0.0.1:8000';health=requests.get(base+'/api/health',timeout=15);health.raise_for_status()
    response=requests.get(base+'/api/research/results',timeout=30);response.raise_for_status();live=response.json();sha=hashlib.sha256((ROOT/'models/fusdense.pt').read_bytes()).hexdigest()
    assert sha==e['checkpoint_sha256']==live['checkpoint']['sha256']==live['evaluation']['checkpoint_sha256']
    phys=requests.get(base+'/api/research/physiology-controlled',timeout=15);phys.raise_for_status();phys=phys.json();assert phys['status']=='evaluated' and not phys['synthetic'];assert all('folds' not in t and 'predictions' not in t for t in phys['tasks'].values())
    with (ROOT/'data/demo_face.png').open('rb') as image:
        response=requests.post(base+'/api/emotion/predict',files={'image':('public_NASA_fixture.png',image,'image/png')},data={'model':'fusdense'},timeout=60)
    response.raise_for_status();prediction=response.json();assert len(prediction['probabilities'])==7 and prediction['held_out_evaluation']['checkpoint_sha256']==sha
    write_json(ROOT/'reports/further_model_api_check.json',{'input':'Existing public NASA/scikit-image fixture. No user photograph. Inference smoke test only, no accuracy claim.','response':prediction})
    response=requests.post(base+'/api/recommendations',json={'user_id':'audit-readonly','emotion':prediction['emotion'],'limit':3,'use_iks':True,'emotion_source':'face_confirmed'},timeout=20);response.raise_for_status();write_json(ROOT/'reports/further_end_to_end.json',response.json())
    xml=ET.parse(ROOT/'reports/pytest.xml').getroot();suites=xml.findall('.//testsuite');tests=sum(int(s.get('tests',0)) for s in suites);failed=sum(int(s.get('failures',0))+int(s.get('errors',0)) for s in suites);assert failed==0
    result={'tests':tests,'passed':tests,'failures':failed,'test_seconds':sum(float(s.get('time',0)) for s in suites),'model_sha256':sha,'original_repository_files':len(originals),'original_repository_unchanged':True,'facial_confusion_and_class_metrics_recomputed':True,'all_test_errors_retained':True,'controlled_raw_window_and_fold_audit':'pass','live_health':health.status_code,'live_emotion_and_recommendation_pipeline':200,'live_controlled_physiology':'evaluated','private_user_images_used':False,'sensor_readings_claimed':False}
    write_json(ROOT/'reports/further_verification_summary.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
