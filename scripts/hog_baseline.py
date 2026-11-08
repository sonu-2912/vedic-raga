"""Fixed HOG + linear SVM comparator, never the deployed FusDenseNet model."""
import argparse,json,time
import cv2
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from threadpoolctl import threadpool_limits
from backend.catalog import ROOT
from backend.datasets import FER2013
from backend.models import FER_CLASSES

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);a=p.parse_args();start=time.perf_counter()
    manifest=json.loads((ROOT/'reports/fer_training_manifest.json').read_text());test_manifest=json.loads((ROOT/'reports/baseline_re_evaluation_manifest.json').read_text())
    hog=cv2.HOGDescriptor((48,48),(16,16),(8,8),(8,8),9);cv2.setNumThreads(1)
    def extract(split,records):
        ds=FER2013(a.data,split);allowed={r['file'] for r in records};ds.rows=[r for r in ds.rows if str(r[0]) in allowed]
        if len(ds)!=len(allowed):raise ValueError('Manifest/data mismatch')
        features=[];labels=[]
        for i in range(len(ds)):
            arr,label=ds.raw_image(i);features.append(hog.compute(cv2.resize(arr,(48,48))).reshape(-1));labels.append(label)
        print(json.dumps({'split':split,'n':len(ds),'features':len(features[0])}),flush=True)
        return np.asarray(features),np.asarray(labels),[str(f) for f,_ in ds.rows]
    train_x,train_y,_=extract('train',manifest['train']);val_x,val_y,_=extract('val',manifest['val']);test_x,test_y,files=extract('test',test_manifest['records'])
    model=make_pipeline(StandardScaler(),LinearSVC(C=.1,class_weight='balanced',dual=False,max_iter=5000,random_state=42))
    with threadpool_limits(limits=2):model.fit(train_x,train_y);val_pred=model.predict(val_x);prediction=model.predict(test_x)
    cm=confusion_matrix(test_y,prediction,labels=list(range(7)))
    report={'model':'Fixed HOG + linear SVM baseline, not FusDenseNet','synthetic':False,'hyperparameters':{'descriptor':'48x48 window, 16x16 block, 8x8 stride/cell, 9 orientation bins','features':int(train_x.shape[1]),'C':.1,'class_weight':'balanced','max_iter':5000,'selection':'Fixed beforehand; no validation or test hyperparameter search'},'train_n':len(train_y),'val_n':len(val_y),'n':len(test_y),'accuracy':accuracy_score(test_y,prediction),'validation_accuracy':accuracy_score(val_y,val_pred),'classification_report':classification_report(test_y,prediction,labels=list(range(7)),target_names=FER_CLASSES,output_dict=True,zero_division=0),'confusion_matrix':cm.tolist(),'seconds':time.perf_counter()-start,'test_files':files,'predictions':{'truth':test_y.tolist(),'predicted':prediction.tolist()},'limitations':'Single fixed classical baseline. No subject-disjoint protocol; exact same final test exclusions as deep models. Test performance is not used for FusDenseNet selection.'}
    (ROOT/'reports/hog_baseline.json').write_text(json.dumps(report,indent=2))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig,ax=plt.subplots(figsize=(8,7));ConfusionMatrixDisplay(cm,display_labels=FER_CLASSES).plot(ax=ax,colorbar=False,cmap='Greys');ax.set_title(f'Actual fixed HOG + linear SVM baseline (n={len(test_y)})');fig.tight_layout();fig.savefig(ROOT/'reports/hog_baseline.png',dpi=150)
    print(json.dumps({k:v for k,v in report.items() if k not in {'test_files','predictions'}},indent=2),flush=True)

if __name__=='__main__':main()
