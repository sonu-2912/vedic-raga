"""Train only on collected SQLite feedback. No artificial negative labels."""
import argparse,json
import torch
from backend.models import FDCF,FER_CLASSES
from backend.catalog import ROOT,RAGAS
from backend.storage import init_db,feedback_rows
from backend.recommender import preference_vectors

def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=80);p.add_argument('--output',default='models/fdcf.pt');p.add_argument('--synthetic-smoke',action='store_true');a=p.parse_args()
    if a.synthetic_smoke and a.output=='models/fdcf.pt':a.output='reports/fdcf_synthetic_smoke.pt'
    init_db();rows=[r for r in feedback_rows() if r['liked']!=0]
    if len(rows)<4 or len(set(r['liked'] for r in rows))<2: raise SystemExit('Pending: at least four explicit like/dislike feedback events with both classes are required. No model saved.')
    users=sorted(set(r['user_id'] for r in rows));torch.manual_seed(42);torch.set_num_threads(2);model=FDCF(len(users),len(RAGAS));opt=torch.optim.Adam(model.parameters(),lr=.005,weight_decay=.001)
    u=torch.tensor([users.index(r['user_id']) for r in rows]);m=torch.tensor([r['raga_id'] for r in rows]);e=torch.tensor([FER_CLASSES.index(r['emotion']) for r in rows]);y=torch.tensor([float(r['liked']==1) for r in rows])
    # Use only preceding observations in each event's preference vector.
    for epoch in range(a.epochs):
        model.train();opt.zero_grad();short=[];long=[]
        for i,r in enumerate(rows):
            s,l=preference_vectors(model,[x for x in rows[:i] if x['user_id']==r['user_id']]);short.append(s);long.append(l)
        logits=model(u,m,e,torch.cat(short),torch.cat(long));loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y);loss.backward();opt.step()
    out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);torch.save({'state_dict':model.state_dict(),'users':users,'observations':len(rows),'synthetic':a.synthetic_smoke},out)
    report={'observations':len(rows),'users':len(users),'training_bce':loss.item(),'epochs':a.epochs,'synthetic':a.synthetic_smoke,'validation':'No held-out ranking evaluation. Training loss is not recommendation quality.'}
    report_name='fdcf_synthetic_training.json' if a.synthetic_smoke else 'fdcf_training.json'
    (ROOT/'reports'/report_name).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
