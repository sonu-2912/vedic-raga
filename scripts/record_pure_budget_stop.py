from pathlib import Path
import json,torch
from scripts.colab_common import write_json
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'reports/pure_fusdense_20261108'
if not (RUN/'completion.json').exists():
 s=torch.load(RUN/'latest.pt',map_location='cpu',weights_only=True);write_json(RUN/'completion.json',{'completed_additional_epochs':len(s['history']),'validation_improved':s['improved'],'test_accessed':False,'external_encoder':False,'teacher':False,'training_only_projection_removed_from_export':True,'synthetic':False,'budget_limited':True,'planned_maximum_epochs':12,'reason':'CPUbudget reallocated to native residual dense variant after completed checkpoint; any partial next epoch discarded and not counted.'})
