"""Keep regressions visible alongside gains; do not modify the frozen export."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
current=json.loads((root/'reports/emotion_evaluation.json').read_text())
backups=sorted((root/'reports/colab_import_backups').glob('*/reports/emotion_evaluation.json'))
before=next(json.loads(p.read_text()) for p in reversed(backups) if json.loads(p.read_text())['checkpoint_sha256']!=current['checkpoint_sha256'])
assert before['predictions']['truth']==current['predictions']['truth']
comparison=json.loads((root/'reports/colab_comparison.json').read_text())
for name,key in [('macro_precision','precision'),('macro_recall','recall')]:
    a=before['classification_report']['macro avg'][key];b=current['classification_report']['macro avg'][key]
    comparison['metrics'][name]={'baseline':a,'colab':b,'difference':b-a}
comparison['baseline_checkpoint_sha256']=before['checkpoint_sha256']
(root/'reports/emotion_training_comparison.json').write_text(json.dumps(comparison,indent=2))
pilot_path=root/'reports/model_comparison.json'
pilot=json.loads(pilot_path.read_text())
old=root/'reports/previous_model_comparison.json'
if not old.exists():old.write_bytes(pilot_path.read_bytes())
pilot['revised']={'accuracy':current['accuracy'],'macro_f1':current['classification_report']['macro avg']['f1-score'],'checkpoint_sha256':current['checkpoint_sha256']}
pilot['accuracy_gain_percentage_points']=100*(current['accuracy']-pilot['baseline']['accuracy'])
pilot['macro_f1_gain']=current['classification_report']['macro avg']['f1-score']-pilot['baseline']['macro_f1']
pilot_path.write_text(json.dumps(pilot,indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
baseline=json.loads((root/'reports/baseline_re_evaluation.json').read_text())
assert baseline['predictions']['truth']==current['predictions']['truth']
labels=['angry','disgust','fear','happy','sad','surprise','neutral'];x=np.arange(7)
fig,axes=plt.subplots(1,2,figsize=(11,4))
for offset,r,label,color in [(-.2,baseline,'Original pilot','#b0957a'),(.2,current,'Current selected model','#244f41')]:
    axes[0].bar(x+offset,[r['classification_report'][c]['recall'] for c in labels],.4,label=label,color=color)
axes[0].set_xticks(x,labels,rotation=30);axes[0].set_ylim(0,1);axes[0].set_ylabel('Recall');axes[0].legend(fontsize=8)
bars=axes[1].bar(['Pilot accuracy','Current accuracy','Pilot macro F1','Current macro F1'],[baseline['accuracy'],current['accuracy'],baseline['classification_report']['macro avg']['f1-score'],current['classification_report']['macro avg']['f1-score']],color=['#b0957a','#244f41']*2)
axes[1].bar_label(bars,fmt='%.3f');axes[1].set_ylim(0,1);axes[1].tick_params(axis='x',labelrotation=20)
fig.suptitle('Same 6,561 retained test faces; full unfiltered metrics');fig.tight_layout();fig.savefig(root/'reports/model_comparison.png',dpi=160);plt.close(fig)
print(json.dumps(comparison['metrics'],indent=2))
