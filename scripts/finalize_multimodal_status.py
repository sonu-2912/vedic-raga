"""Record the completed multimodal software verification without quality claims."""
import json
import shutil
from datetime import datetime
from pathlib import Path

root = Path(__file__).resolve().parents[1]
receipt = json.loads((root / 'reports/slides_build/validation-multimodal.json').read_text())
assert receipt['packageIntegrity']['slide_count'] == 13
assert receipt['packageIntegrity']['findingCount'] == 0
assert receipt['presentationLayout']['findingCount'] == 0
shutil.copy2(root / 'output/RagaMoodSync_Research_Multimodal.pptx', root / 'output/RagaMoodSync_Research.pptx')
status_path = root / 'reports/finalization_status.json'
status = json.loads(status_path.read_text())
status.update(updated=datetime.now().isoformat(), tests_passed=29, tests_total=29,
              powerpoint_sha256=receipt['finalSha256'],
              powerpoint_path='output/RagaMoodSync_Research_Multimodal.pptx',
              visual_review='All 13 rendered slides individually reviewed; contextual recommendation UI verified at desktop and 390px mobile; native PowerPoint execution not claimed',
              multimodal_integration='Implemented: weather, referenced EEG/HRV, missing-modality safeguards, confirmed feeling and listening goal',
              multimodal_outcome_validation='Not performed; synthetic functional ablations only',
              face_disgust_fixed=False)
status_path.write_text(json.dumps(status, indent=2) + '\n')
for name in ['PROJECT_STATUS.md', 'EXPERIMENTS.md', 'MULTIMODAL_STATUS.md']:
    path = root / name
    text = path.read_text(encoding='utf-8')
    marker = '\n## Updated presentation and interface verification\n'
    if marker in text:
        text = text.split(marker)[0]
    text += marker + '\nCurrent deck: `output/RagaMoodSync_Research_Multimodal.pptx` (also copied to `output/RagaMoodSync_Research.pptx`). Exactly 13 editable slides, all rendered slides visually reviewed; package/layout/import/native-chart checks passed with zero findings. Previous FER-only deck is preserved as `output/RagaMoodSync_Research_Updated.pptx`. Native PowerPoint execution is not claimed. Desktop context explanations and 390px mobile layout were checked with isolated test data; no production feedback was written.\n'
    path.write_text(text, encoding='utf-8')
print(json.dumps({'slides': 13, 'tests': 29, 'sha256': receipt['finalSha256']}))
