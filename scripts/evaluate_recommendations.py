"""Chronological evaluation without global-model future-interaction leakage.

    Evaluates the transparent cold-start scorer. FDCF requires a separately trained
    historical checkpoint and is deliberately excluded here.
"""
import json,math,tempfile,os
from backend.catalog import ROOT
from backend.storage import init_db,feedback_rows,connection
from backend.recommender import recommend

def main():
    init_db();rows=feedback_rows();groups={u:[r for r in rows if r['user_id']==u] for u in set(r['user_id'] for r in rows)}
    if not groups or sum(len(v)>=6 for v in groups.values())==0:
        report={'status':'pending','reason':'Need at least six chronological feedback events per evaluable user, including held-out likes. No valid interaction dataset supplied.','events':len(rows)}
    else:
        # Never load a checkpoint trained on held-out data.
        import backend.recommender as rec
        original=rec.load_fdcf;rec.load_fdcf=lambda:None
        old=os.environ.get('RAGAMOOD_DB');metrics={}
        try:
            for use_iks in [True,False]:
                hits=[];ndcgs=[]
                with tempfile.TemporaryDirectory() as tmp:
                    os.environ['RAGAMOOD_DB']=tmp+'/eval.sqlite3';init_db()
                    for user,events in groups.items():
                        if len(events)<6:continue
                        for i,event in enumerate(events):
                            if i>=4 and event['liked']==1:
                                ranked=recommend(user,event['emotion'],limit=10,use_iks=use_iks,persist=False)['recommendations'];ids=[x['id'] for x in ranked]
                                rank=ids.index(event['raga_id'])+1 if event['raga_id'] in ids else None
                                hits.append(float(rank is not None));ndcgs.append(1/math.log2(rank+1) if rank else 0.)
                            with connection() as db:db.execute('INSERT INTO feedback(user_id,raga_id,emotion,liked,mood_improvement) VALUES(?,?,?,?,?)',(user,event['raga_id'],event['emotion'],event['liked'],event['mood_improvement']))
                metrics['iks_on' if use_iks else 'iks_off']={'events':len(hits),'hit_rate_at_10':sum(hits)/len(hits) if hits else None,'ndcg_at_10':sum(ndcgs)/len(ndcgs) if ndcgs else None}
            report={'status':'actual collected feedback evaluation','metrics':metrics,'protocol':'Rolling chronological, 4-event warmup, all catalog candidates, held-out likes only; exposure bias and repeat items remain; no causal efficacy inference.'}
        finally:
            rec.load_fdcf=original
            if old is None:os.environ.pop('RAGAMOOD_DB',None)
            else:os.environ['RAGAMOOD_DB']=old
    (ROOT/'reports/recommendation_evaluation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
