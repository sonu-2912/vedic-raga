from contextlib import asynccontextmanager
from typing import Literal
import json
from fastapi import FastAPI,HTTPException,UploadFile,File,Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field,ConfigDict
from .catalog import ROOT,RAGAS,detail
from .storage import init_db,connection
from .recommender import recommend,fdcf_path
from .emotion import predict
from .signals import eeg_features,hrv_features,sample_signals,fuse_signals
from .context import body_context
from .weather import city_weather
import hashlib
import io
import torch

Emotion=Literal['angry','disgust','fear','happy','sad','surprise','neutral']

@asynccontextmanager
async def lifespan(app):
    init_db()
    yield

app=FastAPI(title='RagaMoodSync',version='1.0.0',lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:5173','http://127.0.0.1:5173'],allow_methods=['GET','POST','DELETE'],allow_headers=['Content-Type'])

@app.middleware('http')
async def fresh_application_document(request,call_next):
    response=await call_next(request)
    if request.url.path in {'/','/index.html'} or request.url.path.startswith('/api/'):
        response.headers['Cache-Control']='no-store'
    return response

class UserInput(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    user_id:str=Field(default='local-user',min_length=1,max_length=80,pattern=r'^[a-zA-Z0-9_-]+$')

class EEGInput(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    samples:list=Field(max_length=131072)
    sample_rate:float=Field(default=128,ge=80,le=4096)
    sample_data:bool=False
    session_id:str|None=Field(default=None,max_length=80)

class HRVInput(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    rr_ms:list[float]|None=Field(default=None,max_length=100000)
    signal:list[float]|None=Field(default=None,max_length=1000000)
    sample_rate:float=Field(default=250,ge=50,le=4096)
    kind:Literal['ecg','ppg']='ecg'
    sample_data:bool=False
    session_id:str|None=Field(default=None,max_length=80)

class WeatherInput(BaseModel):
    model_config=ConfigDict(extra='ignore',allow_inf_nan=False)
    condition:Literal['clear','cloudy','rain','snow','fog','hot','cold']
    temperature_c:float|None=Field(default=None,ge=-90,le=65)
    humidity_percent:float|None=Field(default=None,ge=0,le=100)
    precipitation_mm:float|None=Field(default=None,ge=0,le=1000)
    city:str=Field(default='',max_length=180)
    observed_at:str=Field(default='',max_length=60)
    source:Literal['manual','open_meteo']='manual'

class BodyBaseline(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    rmssd_ms:float|None=Field(default=None,gt=0,le=1000)
    alpha_beta_ratio:float|None=Field(default=None,gt=0,le=10000)
    sample_data:bool=False

class RecommendInput(UserInput):
    emotion:Emotion='neutral'
    limit:int=Field(default=6,ge=1,le=30)
    prahara:int|None=Field(default=None,ge=1,le=8)
    ritu:Literal['vasanta','grishma','varsha','sharad','hemanta','shishira']|None=None
    style:Literal['both','vocal','instrumental']='both'
    use_iks:bool=True
    eeg:EEGInput|None=None
    hrv:HRVInput|None=None
    weather:WeatherInput|None=None
    baseline:BodyBaseline|None=None
    use_sample_context:bool=False
    listening_goal:Literal['balance','calm','energize']='balance'
    emotion_source:Literal['manual','face_confirmed']='manual'

class FeedbackInput(UserInput):
    raga_id:int=Field(ge=0,lt=len(RAGAS))
    emotion:Emotion
    liked:Literal[-1,0,1]=0
    mood_improvement:int=Field(default=0,ge=-2,le=2)

class ListenInput(UserInput):
    raga_id:int=Field(ge=0,lt=len(RAGAS))
    url:str=Field(max_length=300)

@app.get('/api/health')
def health():
    with connection() as db: db.execute('SELECT 1')
    return {'status':'ok','device':'cpu','ragas':len(RAGAS),'link_entries':sum(len(v) for r in RAGAS for v in r['youtube_links'].values()),'models':{'navrasa':'supplied pretrained, native held-out validation unavailable','fusdense':'project-trained checkpoint present; see evaluation report' if (ROOT/'models/fusdense.pt').exists() else 'architecture ready, real training pending','fdcf':'checkpoint present' if fdcf_path().exists() else 'trainable, cold-start heuristic active'},'physical_sensors':False}

@app.get('/api/ragas')
def ragas(q:str=''):
    return [detail(i) for i,r in enumerate(RAGAS) if q.lower() in r['raga_name'].lower()]

@app.get('/api/research/results')
def research_results():
    """Expose measured evidence only for the checkpoint that produced it."""
    path=ROOT/'models/fusdense.pt'
    if not path.exists():return {'status':'untrained','evaluation':None}
    checkpoint_bytes=path.read_bytes()
    ck=torch.load(io.BytesIO(checkpoint_bytes),map_location='cpu',weights_only=True)
    digest=hashlib.sha256(checkpoint_bytes).hexdigest()
    report=ROOT/'reports/emotion_evaluation.json'
    evidence=json.loads(report.read_text()) if report.exists() else None
    matched=evidence is not None and evidence.get('checkpoint_sha256')==digest
    history_path=ROOT/'reports/training_history.json'
    try:history=json.loads(history_path.read_text()) if history_path.exists() else []
    except json.JSONDecodeError:history=[] # An epoch may be publishing its new history.
    comparison_path=ROOT/'reports/model_comparison.json'
    comparison=json.loads(comparison_path.read_text()) if comparison_path.exists() else None
    if comparison and comparison.get('revised',{}).get('checkpoint_sha256')!=digest:comparison=None
    continuation_path=ROOT/'reports/emotion_training_comparison.json'
    continuation=json.loads(continuation_path.read_text()) if continuation_path.exists() else None
    if continuation and continuation.get('checkpoint_sha256')!=digest:continuation=None
    experiment_path=ROOT/'reports/latest_emotion_experiment.json'
    experiment=json.loads(experiment_path.read_text()) if experiment_path.exists() else None
    if experiment:
        experiment['candidate_is_active']=experiment.get('candidate_checkpoint_sha256')==digest
    audit_path=ROOT/'reports/bottleneck_audit_20261108/audit.json'
    bottleneck_audit=json.loads(audit_path.read_text()) if audit_path.exists() else None
    return {'status':'evaluated' if matched else 'evaluation_pending','checkpoint':{'sha256':digest,'pure_fusdense':not any(k.startswith('transfer_') for k in ck['state_dict']),'external_encoder':any(k.startswith('transfer_') for k in ck['state_dict']),'epoch':ck.get('epoch'),'train_count':ck.get('train_count'),'val_count':ck.get('val_count'),'architecture':ck.get('architecture',{}),'inference':ck.get('inference',{}),'selection_metric':ck.get('selection_metric','validation accuracy'),'head_refinement':bool(ck.get('head_refinement')),'training_stage':ck.get('training_stage','classifier_refinement' if ck.get('head_refinement') else 'full_network'),'parameters':sum(v.numel() for k,v in ck['state_dict'].items() if not k.endswith(('running_mean','running_var','num_batches_tracked')))},'evaluation':{k:evidence[k] for k in ['n','accuracy','classification_report','confusion_matrix','checkpoint_epoch','checkpoint_sha256','split','negative_log_likelihood','expected_calibration_error_10_bins','brier_score'] if k in evidence} if matched else None,'comparison':comparison,'continuation_comparison':continuation,'latest_experiment':experiment,'bottleneck_audit':bottleneck_audit,'history':history,'limitations':['Paper-informed implementation; layer dimensions and spatial pooling are project choices.','FER2013 labels measure expressions, not a personâ€™s internal mood.','No clinical validation or identity-disjoint benchmark.']}

@app.get('/api/research/plots/{name}')
def research_plot(name:str):
    from fastapi.responses import FileResponse
    allowed={'repair':'pure_orientation_validation.png','repair_errors':'pure_orientation_validation_confusion.png','confusion':'emotion_evaluation.png','training':'training_curves.png','comparison':'model_comparison.png','experiment':'spatial_generalization_trials.png','pure_experiment':'pure_fusdense_training.png','pure_losses':'pure_fusdense_losses.png','pure_validation':'pure_fusdense_validation.png','physiology':'gamer_fusion.png','physiology_controlled':'gamer_fixed_context.png'}
    if name not in allowed or not (ROOT/'reports'/allowed[name]).exists():raise HTTPException(404,'Plot unavailable')
    return FileResponse(ROOT/'reports'/allowed[name],media_type='image/png',headers={'Cache-Control':'no-store'})

@app.get('/api/research/physiology')
def physiology_evidence():
    path=ROOT/'reports/gamer_evaluation.json'
    if not path.exists():return {'status':'pending','synthetic':False,'tasks':{},'message':'Real GamER evaluation has not completed. No result is claimed.'}
    result=json.loads(path.read_text())
    # Do not expose participant-level annotation histories through the app API.
    tasks={name:{k:v for k,v in value.items() if k not in {'predictions','folds'}} for name,value in result['tasks'].items()}
    sensitivity=ROOT/'reports/gamer_evaluation_plus_offset.json'
    alternate=json.loads(sensitivity.read_text()) if sensitivity.exists() else None
    return {'status':'evaluated','synthetic':False,'protocol':result['protocol'],'tasks':tasks,
            'accepted_intervals':result['accepted_intervals'],'exclusion_count':result['exclusion_count'],
            'offset_sign':result['offset_sign'],'sensitivity_available':sensitivity.exists(),
            'offset_sensitivity':{name:{'n_intervals':value['n_intervals'],'n_subjects':value['n_subjects'],'metrics':{method:{'accuracy':metrics['accuracy'],'macro_f1':metrics['macro_f1']} for method,metrics in value['results'].items()}} for name,value in alternate['tasks'].items()} if alternate else None,
            'report_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

@app.get('/api/research/physiology-controlled')
def controlled_physiology_evidence():
    path=ROOT/'reports/gamer_fixed_context_evaluation.json'
    if not path.exists():return {'status':'pending','tasks':{}}
    report=json.loads(path.read_text())
    tasks={name:{k:v for k,v in task.items() if k not in ['predictions','folds']} for name,task in report['tasks'].items()}
    return {'status':'evaluated','synthetic':False,'protocol':report['protocol'],'tasks':tasks,'report_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

@app.get('/api/ragas/{raga_id}')
def raga(raga_id:int):
    if not 0<=raga_id<len(RAGAS): raise HTTPException(404,'Raga not found')
    return detail(raga_id)

@app.post('/api/emotion/predict')
def emotion(image:UploadFile=File(...),model:Literal['navrasa','fusdense']=Form('fusdense'),denoising:Literal['none','nlm','bilateral','gaussian','hybrid']=Form('none'),require_face:bool=Form(True)):
    raw=image.file.read(8*1024*1024+1)
    try: return predict(raw,model,denoising,require_face)
    except ValueError as e: raise HTTPException(422,str(e))
    except RuntimeError as e: raise HTTPException(503,str(e))

@app.post('/api/recommendations')
def recommendations(payload:RecommendInput):
    try:
        eeg=eeg_features(payload.eeg.samples,payload.eeg.sample_rate,payload.eeg.sample_data) if payload.eeg else None
        hrv=hrv_features(**{'rr_ms':payload.hrv.rr_ms,'signal':payload.hrv.signal,'sample_rate':payload.hrv.sample_rate,'kind':payload.hrv.kind,'sample':payload.hrv.sample_data}) if payload.hrv else None
        if eeg:eeg['session_id']=payload.eeg.session_id
        if hrv:hrv['session_id']=payload.hrv.session_id
        body=body_context(eeg,hrv,payload.baseline.model_dump() if payload.baseline else None,payload.use_sample_context) if eeg or hrv else None
        return recommend(payload.user_id,payload.emotion,payload.limit,payload.prahara,payload.ritu,payload.style,payload.use_iks,body,weather=payload.weather.model_dump() if payload.weather else None,listening_goal=payload.listening_goal,emotion_source=payload.emotion_source)
    except ValueError as e: raise HTTPException(422,str(e))

@app.post('/api/feedback')
def feedback(p:FeedbackInput):
    with connection() as db:
        cur=db.execute('INSERT INTO feedback(user_id,raga_id,emotion,liked,mood_improvement) VALUES(?,?,?,?,?)',(p.user_id,p.raga_id,p.emotion,p.liked,p.mood_improvement))
        return {'id':cur.lastrowid,'saved':True,'preference_update':'Immediate short/long preference update. FDCF retraining is explicit: python -m scripts.train_fdcf'}

@app.post('/api/listening')
def listening(p:ListenInput):
    links=[v for group in RAGAS[p.raga_id]['youtube_links'].values() for v in group]
    if p.url not in links: raise HTTPException(422,'URL must be from this raga catalog')
    with connection() as db:
        cur=db.execute('INSERT INTO listening(user_id,raga_id,url) VALUES(?,?,?)',(p.user_id,p.raga_id,p.url))
        return {'saved':True,'id':cur.lastrowid,'meaning':'Link opened; playback completion is not measured'}

@app.get('/api/history/{user_id}')
def history(user_id:str):
    with connection() as db:
        out={key:[dict(r) for r in db.execute(f'SELECT * FROM {key} WHERE user_id=? ORDER BY id DESC LIMIT 100',(user_id,))] for key in ['feedback','listening','recommendations']}
    for group in out.values():
        for row in group: row['raga_name']=RAGAS[row['raga_id']]['raga_name']
    return out

@app.delete('/api/history/{user_id}')
def delete_history(user_id:str):
    with connection() as db:
        for table in ['feedback','listening','recommendations']: db.execute(f'DELETE FROM {table} WHERE user_id=?',(user_id,))
    # Existing trained models may retain data: disable cached learned checkpoint on erasure.
    path=fdcf_path()
    if path.exists():
        import time
        path.rename(path.with_name('fdcf.disabled-after-erasure-'+str(time.time_ns())+'.pt'))
    return {'deleted':True,'model_disabled':True}

@app.get('/api/signals/sample')
def samples(profile:Literal['resting','activated']='resting'):
    result=sample_signals()
    result['eeg']['session_id']=result['hrv']['session_id']='synthetic-demo'
    if profile=='activated':
        import numpy as np
        rng=np.random.default_rng(67);t=np.arange(128*16)/128
        result['eeg']['samples']=[(4e-6*np.sin(2*np.pi*10*t)+15e-6*np.sin(2*np.pi*20*t)+2e-6*rng.normal(size=len(t))).tolist()]
    result['baseline']={'rmssd_ms':30.,'alpha_beta_ratio':3.,'sample_data':True}
    result['disclosure']='Synthetic demonstration, including an illustrative simulated reference. No readings or personal baseline were collected.'
    return result

@app.get('/api/context/weather')
def weather(city:str):
    try:return city_weather(city)
    except ValueError as e:raise HTTPException(422,str(e))
    except Exception as e:raise HTTPException(502,'Weather provider unavailable. Enter weather manually; other inputs still work.') from e

@app.post('/api/eeg/process')
def eeg(p:EEGInput):
    try: return eeg_features(p.samples,p.sample_rate,p.sample_data)
    except (ValueError,TypeError) as e: raise HTTPException(422,str(e))

@app.post('/api/hrv/process')
def hrv(p:HRVInput):
    try: return hrv_features(p.rr_ms,p.signal,p.sample_rate,p.kind,p.sample_data)
    except (ValueError,TypeError,IndexError) as e: raise HTTPException(422,str(e))

# A built frontend also runs from one local backend server.
dist=ROOT/'frontend/dist'
if dist.exists(): app.mount('/',StaticFiles(directory=dist,html=True),name='frontend')
