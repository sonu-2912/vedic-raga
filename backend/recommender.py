"""Transparent IKS cold start + observed preferences + optional trained FDCF."""
import math,os
from pathlib import Path
import torch
from .catalog import ROOT,RAGAS,BHAVAS,detail
from .models import FDCF,FER_CLASSES
from .storage import feedback_rows,connection
from .context import weather_context

_cached = None

def fdcf_path():
    return Path(os.environ.get('RAGAMOOD_FDCF',str(ROOT/'models/fdcf.pt')))

def load_fdcf():
    global _cached
    path=fdcf_path()
    if not path.exists():
        return None
    stamp=path.stat().st_mtime_ns
    if _cached and _cached[0]==stamp:
        return _cached[1]
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck.get('synthetic',False): return None
    model=FDCF(len(ck['users']),len(RAGAS))
    model.load_state_dict(ck['state_dict']); model.eval()
    value=(model,ck['users'])
    _cached=(stamp,value)
    return value

def preference_vectors(model, rows):
    def vec(items):
        if not items:
            return torch.zeros(1,model.music.embedding_dim)
        ids=torch.tensor([r['raga_id'] for r in items])
        weights=torch.tensor([r['liked']+.25*r['mood_improvement'] for r in items],dtype=torch.float32)
        return (model.music(ids)*weights[:,None]).sum(0,keepdim=True)/weights.abs().sum().clamp_min(1)
    return vec(rows[-5:]),vec(rows)

def recommend(user,emotion,limit=6,prahara=None,ritu=None,style='both',use_iks=True,physiology=None,persist=True,weather=None,listening_goal='balance',emotion_source='manual'):
    rows=feedback_rows(user)
    effective_goal='calm' if listening_goal=='balance' and physiology and physiology.get('calming_context') else listening_goal
    target='shama' if effective_goal=='calm' else 'utsaha' if effective_goal=='energize' else BHAVAS[emotion]
    weather_info=weather_context(weather)
    trained=load_fdcf()
    neural={}
    if trained and user in trained[1]:
        model,users=trained
        with torch.inference_mode():
            short,long=preference_vectors(model,rows)
            n=len(RAGAS)
            scores=model(torch.full((n,),users.index(user)),torch.arange(n),torch.full((n,),FER_CLASSES.index(emotion)),short.expand(n,-1),long.expand(n,-1)).sigmoid().tolist()
            neural=dict(enumerate(scores))
    results=[]
    for i,r in enumerate(RAGAS):
        matches=[r['primary_sthayi_bhava'].lower(),*[x.lower() for x in r['secondary_sthayi_bhava']]]
        bhava=1. if target==matches[0] else (.6 if target in matches else 0.)
        time=float(prahara in r['prahara']) if prahara else .5
        season=float(ritu.lower()==r['ritu'].lower()) if ritu and r['ritu'] else .5
        iks=.65*bhava+.2*time+.15*season if use_iks else .5
        def preference(items):
            selected=[x for x in items if x['raga_id']==i]
            # Shrink sparse feedback, avoid unlimited repeated-vote growth.
            return sum(x['liked']+.25*x['mood_improvement'] for x in selected)/(len(selected)+2)
        short,long=preference(rows[-5:]),preference(rows)
        beta=.7 if rows else .5
        pref=beta*short+(1-beta)*long
        # Optional physiology is a bounded engineering heuristic, no affect classifier.
        calming=0.
        if physiology and physiology.get('calming_context') and effective_goal=='calm' and 'shama' in matches:
            calming=.03 if matches[0]=='shama' else .015
        weather_adjustment=.03 if use_iks and (r.get('ritu') or '').lower() in weather_info['season_tags'] else 0.
        base=.65*iks+.35*(.5+.5*math.tanh(pref))+calming+weather_adjustment
        score=.75*base+.25*neural[i] if i in neural else base
        d=detail(i)
        d['youtube_links']={k:v for k,v in r['youtube_links'].items() if style=='both' or k==style}
        if not any(d['youtube_links'].values()):
            continue
        d.update(score=round(score,6),components={'iks':round(iks,4),'bhava':bhava,'short_preference':short,'long_preference':long,'beta':beta,'physiology_adjustment':calming,'weather_adjustment':weather_adjustment,'fdcf':neural.get(i)},reason=f"Target bhava: {target}; listening goal: {effective_goal}. Metadata is source-supplied. Preference observations: {len(rows)}.")
        results.append(d)
    results.sort(key=lambda r:(-r['score'],r['raga_name']))
    results=results[:limit]
    mode='trained FDCF hybrid (unvalidated)' if neural else 'IKS + feedback cold-start heuristic (FDCF not trained for this user)'
    if persist:
        with connection() as db:
            db.executemany('INSERT INTO recommendations(user_id,emotion,raga_id,score,mode) VALUES(?,?,?,?,?)',[(user,emotion,r['id'],r['score'],mode) for r in results])
    return {'emotion':emotion,'target_bhava':target,'ranking_mode':mode,'feedback_observations':len(rows),'recommendations':results,'modalities':{'face_or_manual_emotion':True,'physiology':bool(physiology),'weather':weather_info['available'],'iks':use_iks,'preferences':bool(rows)},'context':{'emotion_source':emotion_source,'requested_goal':listening_goal,'effective_goal':effective_goal,'weather':weather_info,'physiology':physiology,'priority':'Confirmed mood and explicit listening goal first; body reference policy and weather add experimental context. No emotion is inferred from weather or physiological features.','fusion_validation':'Rule-based engineering integration; no measured multimodal recognition or mood-improvement benefit.'},'disclaimer':'Research prototype. Raga associations and inferred expressions do not establish therapeutic efficacy.'}
