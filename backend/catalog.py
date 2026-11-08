import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'external/raag-ratnakar-main/raga_sthayi_bhava_dataset_enriched.json'
if not SOURCE.exists():raise RuntimeError('Raga catalog is missing. Run python -m scripts.bootstrap_resources --download or provide your inspected local source directory with --source-dir. See THIRD_PARTY_NOTICES.md.')
RAGAS = json.loads(SOURCE.read_text(encoding='utf-8'))
BHAVAS = {'angry':'krodha','disgust':'shama','fear':'bhaya','happy':'hasa','sad':'shoka','surprise':'vismaya','neutral':'shama'}
# Adapted from Raag Ratnakar rasa_bhava_mapper.py. Disgust fallback is a policy.
RASA_EMOTIONS = {'adhbuta':'surprise','bhayanak':'fear','bhibhatsa':'disgust','hasya':'happy','karuna':'sad','raudra':'angry','shanta':'neutral','shringara':'happy','veer':'happy'}

def detail(index):
    r = dict(RAGAS[index])
    r['id'] = index
    r['metadata_provenance'] = 'Raag Ratnakar supplied JSON; not independently musicologically verified'
    r['metadata_status'] = {k:('source supplied, unverified' if r.get(k) else 'missing') for k in r if k not in {'id','metadata_provenance'}}
    for k in ['thaat','aaroh','avaroh','vadi','samvadi']:
        r[k] = None
        r['metadata_status'][k] = 'missing'
    return r
