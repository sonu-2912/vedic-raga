"""Attach measured evidence components without changing existing listener flows."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
def replace(relative,old,new):
    path=root/relative;text=path.read_text(encoding='utf-8')
    if new in text:return
    if old=='<p>These measurements do not establish' and 'Game-context control:' in text:return
    assert old in text,relative
    path.write_text(text.replace(old,new,1),encoding='utf-8')
replace('frontend/src/ListenerApp.jsx',"import './listener.css';","import './listener.css';\nimport PhotoEvidence from './PhotoEvidence.jsx';")
replace('frontend/src/ListenerApp.jsx','{moodButtons(true)}<button','<PhotoEvidence prediction={detected}/>{moodButtons(true)}<button')
replace('frontend/src/research/ResearchStudio.jsx',"import TrainingEvidence from './TrainingEvidence.jsx';","import TrainingEvidence from './TrainingEvidence.jsx';\nimport PhysiologyEvidence from './PhysiologyEvidence.jsx';")
replace('frontend/src/research/ResearchStudio.jsx','<TrainingEvidence/><div','<TrainingEvidence/><PhysiologyEvidence/><div')
replace('frontend/src/research/TrainingEvidence.jsx',"import React,{useEffect,useState} from 'react';","import React,{useEffect,useState} from 'react';\nimport ContinuationEvidence from './ContinuationEvidence.jsx';")
replace('frontend/src/research/TrainingEvidence.jsx','{data.comparison&&','<ContinuationEvidence report={data.continuation_comparison}/>{data.comparison&&')
replace('backend/main.py',"'comparison':comparison,'history':history","'comparison':comparison,'continuation_comparison':continuation,'history':history")
replace('backend/main.py',"'selection_metric':ck.get('selection_metric','validation accuracy'),'parameters'","'selection_metric':ck.get('selection_metric','validation accuracy'),'head_refinement':bool(ck.get('head_refinement')),'parameters'")
replace('frontend/src/research/TrainingEvidence.jsx','<h3>Implemented feature path</h3>',"{c?.head_refinement&&<p><strong>Selected training method:</strong> the existing classifier MLP was refined on frozen FusDenseNet features. Normalization used training features only and was folded into the existing Linear layer. Epoch indices include both full-network and cheaper classifier-only phases; they are not equivalent compute.</p>}<h3>Implemented feature path</h3>")
replace('frontend/src/research/PhysiologyEvidence.jsx','These measurements do not establish seven-class','The game-ID-only control outperforms physiological fusion here, and the confidence intervals for fusion gains include zero. The five-class fusion accuracy is below the majority baseline. These measurements do not establish seven-class')
replace('frontend/src/research/PhysiologyEvidence.jsx','<p>These measurements do not establish',"<p><strong>Game-context control:</strong> Source emotion labels correlate with game identity. The game-ID-only row tests this confound; it is not physiological emotion recognition.</p>{data.offset_sensitivity&&<details><summary>Offset alignment sensitivity</summary>{Object.entries(data.offset_sensitivity).map(([name,t])=><div key={name}><h4>{name.replaceAll('_',' ')}</h4><p>Adding the offset: {t.n_intervals} intervals / {t.n_subjects} participants.</p><ul>{Object.entries(t.metrics).map(([method,m])=><li key={method}>{method}: {(100*m.accuracy).toFixed(2)}% accuracy; {m.macro_f1.toFixed(4)} macro F1</li>)}</ul></div>)}</details>}<p>These measurements do not establish")
