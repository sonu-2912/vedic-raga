"""Preserve the deployed comparison baseline before any further promotion."""
import hashlib,json,shutil
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'reports/further_baseline';out.mkdir(exist_ok=True)
for source,name in [('models/fusdense.pt','fusdense.pt'),('reports/emotion_evaluation.json','emotion_evaluation.json'),('reports/emotion_evaluation_manifest.json','emotion_evaluation_manifest.json'),('reports/training_history.json','training_history.json'),('reports/fer_training_manifest.json','fer_training_manifest.json')]:
    target=out/name
    if not target.exists():shutil.copy2(root/source,target)
sha=hashlib.sha256((out/'fusdense.pt').read_bytes()).hexdigest()
assert json.loads((out/'emotion_evaluation.json').read_text())['checkpoint_sha256']==sha
for name in ['further_fusdense_run','frozen_bn_fusdense_run']:
    settings=root/f'reports/{name}/settings.json'
    if settings.exists():assert json.loads(settings.read_text())['initial_checkpoint_sha256']==sha
print(sha)
