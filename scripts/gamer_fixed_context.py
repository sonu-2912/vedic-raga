"""Game-constant, fixed-30-second real EEG/cardiac analysis, with LOSO."""
import json
from scripts.gamer_experiment import extract
from scripts.gamer_controlled import ROOT,task,write,PROTOCOL

def main():
    protocol={**PROTOCOL,'primary_fixed_window_seconds':30,'window_selection':'Centered 30-second slice fully inside an annotation of at least 30s; identical duration for both labels','primary_tasks':['QWOP: Angry vs Happy','CBAAD: Bored vs Stressed'],'game_identity':'Constant inside each independently fitted/evaluated task','secondary_analysis':'Whole >=15s intervals are exploratory sensitivity only, not conventional HRV'}
    path=ROOT/'reports/gamer_fixed_context_protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Scored protocol changed')
    write('gamer_fixed_context_protocol.json',protocol)
    tasks={}
    for sign,suffix in [(-1,''),(1,'_plus_offset')]:
        cache=ROOT/f'reports/gamer_fixed30_features{suffix}.json'
        features=json.loads(cache.read_text()) if cache.exists() else extract(sign,30)
        assert all(r['quality']['duration_seconds']==30 for r in features['records'])
        assert len(features['records'])==len({r['annotation_id'] for r in features['records']})
        for game,labels in PROTOCOL['tasks'].items():
            tasks[game+suffix]=task(features['records'],game,labels,30)
            print(game+suffix,json.dumps({k:v for k,v in tasks[game+suffix].items() if k in ['status','n','participants','reason','fusion_minus_eeg_macro_f1']}),flush=True)
    report={'protocol':protocol,'synthetic':False,'tasks':tasks}
    write('gamer_fixed_context_evaluation.json',report)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,game in zip(axes,PROTOCOL['tasks']):
        t=tasks[game]
        if t['status']!='evaluated':ax.text(.05,.5,t['reason'],wrap=True);ax.set_title(game);continue
        names=['EEG','Cardiac','Fusion','Majority','Duration only'];bars=ax.bar(names,[t['results'][n]['macro_f1'] for n in names],color=['#829269','#b58157','#244f41','#ccc','#9c8ca4']);ax.bar_label(bars,fmt='%.3f');ax.set_ylim(0,1);ax.tick_params(axis='x',labelrotation=20);ax.set_ylabel('Macro F1');ax.set_title(f'{game} constant · {t["n"]} windows / {t["participants"]} people')
    fig.suptitle('Real physiology: game held constant, fixed 30s duration, held-out participants');fig.tight_layout();fig.savefig(ROOT/'reports/gamer_fixed_context.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
