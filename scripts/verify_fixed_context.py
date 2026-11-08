"""Audit fixed context/duration, participant folds and original raw windows."""
import io,json,zipfile
import h5py
import numpy as np
from scripts.gamer_controlled import ROOT
from backend.physiology_features import paired_features

def main():
    report=json.loads((ROOT/'reports/gamer_fixed_context_evaluation.json').read_text());checks=[]
    with zipfile.ZipFile(ROOT/'data/private/gamer/data.zip') as archive:
        for suffix in ['', '_plus_offset']:
            features=json.loads((ROOT/f'reports/gamer_fixed30_features{suffix}.json').read_text());by_id={r['annotation_id']:r for r in features['records']}
            assert len(by_id)==len(features['records'])
            for game in ['QWOP','CBAAD']:
                item=report['tasks'][game+suffix]
                if item['status']!='evaluated':continue
                ids=[i for f in item['folds'] for i in f['annotation_ids']]
                assert len(ids)==len(set(ids))==item['n']
                rows=[by_id[i] for i in ids];assert {r['game'] for r in rows}=={game}
                assert {r['quality']['duration_seconds'] for r in rows}=={30.}
                assert all(r['last_sample']-r['first_sample']==round(30*r['sample_rate']) for r in rows)
                for fold in item['folds']:
                    assert fold['subject'] not in fold['training_subjects']
                    assert all(by_id[i]['subject']==fold['subject'] for i in fold['annotation_ids'])
                assert len(set(item['predictions']['Game only (constant)']))==1
                assert len(set(item['predictions']['Duration only']))==1
                y=np.array(item['predictions']['truth']);labels=item['labels']
                for method,metrics in item['results'].items():
                    p=np.array(item['predictions'][method]);matrix=np.array([[np.sum((y==a)&(p==b)) for b in labels] for a in labels])
                    assert matrix.tolist()==metrics['confusion_matrix']
                    assert abs(np.trace(matrix)/matrix.sum()-metrics['accuracy'])<1e-12
                # Deterministic source audit, not a selected favorable prediction.
                raw=sorted(rows,key=lambda r:r['annotation_id'])[0]
                with h5py.File(io.BytesIO(archive.read(raw['recording'])),'r') as f:
                    device=f[next(iter(f))];channels={str(v.attrs.get('sensor','')):k for k,v in device['raw'].items()};a,b=raw['first_sample'],raw['last_sample']
                    eeg,cardiac,q=paired_features(device['raw'][channels['EEG']][a:b],device['raw'][channels['ECG']][a:b],raw['sample_rate'])
                assert np.allclose(eeg,raw['eeg'],rtol=1e-10,atol=1e-10) and np.allclose(cardiac,raw['cardiac'],rtol=1e-10,atol=1e-10)
                checks.append({'task':game+suffix,'windows':item['n'],'participants':item['participants'],'game_values':1,'window_seconds':30,'participant_overlap':False,'raw_window_recomputed':True,'game_and_duration_controls_constant':True})
    result={'status':'pass','synthetic':False,'checks':checks,'claim_boundary':'Game identity and window duration are controlled; no claim that every possible physiological/annotation confound is eliminated'}
    (ROOT/'reports/gamer_fixed_context_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
