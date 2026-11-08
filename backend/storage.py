import os, sqlite3
from contextlib import contextmanager
from .catalog import ROOT

def db_path():
    return os.environ.get('RAGAMOOD_DB',str(ROOT/'data/ragamood.sqlite3'))

@contextmanager
def connection():
    path=db_path()
    from pathlib import Path
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(path,timeout=15)
    db.row_factory=sqlite3.Row
    db.execute('PRAGMA foreign_keys=ON')
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

def init_db():
    with connection() as db:
        db.executescript('''
        CREATE TABLE IF NOT EXISTS recommendations(
          id INTEGER PRIMARY KEY, user_id TEXT NOT NULL, emotion TEXT NOT NULL,
          raga_id INTEGER NOT NULL, score REAL NOT NULL, mode TEXT NOT NULL,
          created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')));
        CREATE TABLE IF NOT EXISTS feedback(
          id INTEGER PRIMARY KEY, user_id TEXT NOT NULL, raga_id INTEGER NOT NULL,
          emotion TEXT NOT NULL, liked INTEGER CHECK(liked IN (-1,0,1)),
          mood_improvement INTEGER CHECK(mood_improvement BETWEEN -2 AND 2),
          created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')));
        CREATE TABLE IF NOT EXISTS listening(
          id INTEGER PRIMARY KEY,user_id TEXT NOT NULL,raga_id INTEGER NOT NULL,
          url TEXT NOT NULL,created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')));
        CREATE INDEX IF NOT EXISTS feedback_user ON feedback(user_id,created_at);
        ''')

def feedback_rows(user=None):
    with connection() as db:
        rows=db.execute('SELECT * FROM feedback'+(' WHERE user_id=?' if user else '')+' ORDER BY id', (user,) if user else ()).fetchall()
        return [dict(r) for r in rows]
