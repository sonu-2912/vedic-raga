"""Separate OpenCV expression-teacher screen/cache. Never opens a test split."""
import argparse,json,time
from pathlib import Path
import cv2
import numpy as np
import torch
from scripts.colab_common import fixed_split,digest,metrics,write_json,save_checkpoint
ROOT=Path(__file__).resolve().parents[1]

def teacher_blob(images):
    rgb=[cv2.cvtColor(cv2.resize(a,(112,112),interpolation=cv2.INTER_LINEAR),cv2.COLOR_GRAY2BGR) for a in images]
    return cv2.dnn.blobFromImages(rgb,1/127.5,(112,112),(127.5,127.5,127.5),swapRB=True,crop=False)

def fer_order(scores):
    if scores.ndim!=2 or scores.shape[1]!=7:raise ValueError('Expected exactly seven native outputs')
    return scores[:,[0,1,2,3,5,6,4]]

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--split',choices=['train','val'],default='val');a=p.parse_args();torch.set_num_threads(2);cv2.setNumThreads(2)
    folder=ROOT/'reports/teacher_screen_20261107';provenance=json.loads((folder/'provenance.json').read_text());model=ROOT/'external/opencv_fer_teacher/mobilefacenet.onnx';assert digest(model)==provenance['model_sha256']
    out=folder/f'{a.split}_teacher.pt'
    if out.exists():raise RuntimeError('Existing teacher cache preserved; inspect/reuse it.')
    manifest=json.loads((ROOT/'reports/fer_training_manifest.json').read_text());ds=fixed_split(a.data,manifest,a.split)
    net=cv2.dnn.readNet(str(model));net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV);net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    batch=32
    try:
        net.setInput(teacher_blob([ds.raw_image(0)[0],ds.raw_image(1)[0]]),'data');trial=net.forward('label');assert trial.reshape(2,-1).shape==(2,7)
    except (cv2.error,AssertionError,ValueError):batch=1;net=cv2.dnn.readNet(str(model))
    results=[];labels=[];began=time.perf_counter()
    for start in range(0,len(ds),batch):
        end=min(len(ds),start+batch);rows=[ds.raw_image(i) for i in range(start,end)];net.setInput(teacher_blob([r[0] for r in rows]),'data');raw=net.forward('label').reshape(len(rows),7);assert np.isfinite(raw).all();results.append(fer_order(raw.copy()));labels.extend(r[1] for r in rows)
        if (start//batch+1)%25==0:print(json.dumps({'split':a.split,'faces':end,'total':len(ds),'batch':batch}),flush=True)
    scores=np.concatenate(results);probabilities=bool((scores>=0).all() and np.allclose(scores.sum(1),1,rtol=1e-4,atol=1e-5))
    logits=np.log(np.maximum(scores,1e-12)) if probabilities else scores
    tensor=torch.from_numpy(logits.astype(np.float32));y=torch.tensor(labels)
    save_checkpoint(out,{'model_sha256':provenance['model_sha256'],'manifest_sha256':digest(ROOT/'reports/fer_training_manifest.json'),'split':a.split,'logits':tensor,'labels':y,'source_is_probability':probabilities,'training_overlap_unknown':True})
    result=metrics(tensor,y);result.update({'split':a.split,'model_type':'External OpenCV MobileFaceNet, not FusDenseNet','model_sha256':provenance['model_sha256'],'training_overlap_unknown':True,'test_accessed':False,'n':len(ds),'preprocessing':provenance['adaptation'],'native_outputs_were_probabilities':probabilities,'seconds':time.perf_counter()-began,'limitation':'Foreign checkpoint full training manifest unavailable; not an independently clean benchmark, not a FusDenseNet result. Already-cropped FER inputs resized without fabricated landmarks.'})
    write_json(folder/f'{a.split}_evaluation.json',result);print(json.dumps({k:result[k] for k in ['n','split','accuracy','classification_report','model_type','training_overlap_unknown']},indent=2))
if __name__=='__main__':main()
