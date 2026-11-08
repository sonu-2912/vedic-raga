"""Exercise the FDCF training command on synthetic explicit interactions only."""
import os,sys,tempfile
from backend.storage import init_db,connection
from scripts.train_fdcf import main as train

def main():
    old=os.environ.get('RAGAMOOD_DB');args=sys.argv[:]
    try:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ['RAGAMOOD_DB']=tmp+'/synthetic.sqlite3';init_db()
            with connection() as db:
                db.executemany('INSERT INTO feedback(user_id,raga_id,emotion,liked,mood_improvement) VALUES(?,?,?,?,?)',[(f'synthetic-{u}',i,'happy' if i%2 else 'sad',1 if i%2 else -1,1 if i%2 else -1) for u in range(2) for i in range(8)])
            sys.argv=['train_fdcf','--epochs','8','--synthetic-smoke'];train()
    finally:
        sys.argv=args
        if old is None:os.environ.pop('RAGAMOOD_DB',None)
        else:os.environ['RAGAMOOD_DB']=old
if __name__=='__main__':main()
