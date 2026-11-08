"""Record only saved complete epochs after the declared conditional budget transition."""
import json
from pathlib import Path
import torch
from scripts.colab_common import write_json
root=Path(__file__).resolve().parents[1];run=root/'reports/pure_repair_20261108'
if not (run/'completion.json').exists():
 torch.set_num_threads(1);ck=torch.load(run/'latest.pt',map_location='cpu',weights_only=True);rows=ck['history'];assert len(rows)>=6 and not ck['improved'] and not (run/'best.pt').exists();write_json(run/'stage_transition.json',{'completed_epochs':len(rows),'discard_partial_next_epoch':True,'reason':'Conditional budget cap declared after4validation epochs, before test. No candidate qualified after the6epochminimum; allocate remaining CPU to orientation classifier repair.','saved_checkpoint_epoch':rows[-1]['epoch'],'elapsed_wall_seconds_at_last_completed_epoch':ck['elapsed_wall_seconds'],'test_accessed':False});write_json(run/'completion.json',{'completed_additional_epochs':len(rows),'validation_improved':False,'test_accessed':False,'external_encoder':False,'teacher':False,'stop_reason':'conditional_budget_reallocation_after_six_complete_epochs','elapsed_wall_seconds':ck['elapsed_wall_seconds'],'synthetic':False})
