from pathlib import Path
root=Path(__file__).resolve().parents[1]
path=root/'frontend/src/research/ResearchStudio.jsx';text=path.read_text(encoding='utf-8')
if "import ControlledPhysiology" not in text:text=text.replace("import PhysiologyEvidence from './PhysiologyEvidence.jsx';","import PhysiologyEvidence from './PhysiologyEvidence.jsx';\nimport ControlledPhysiology from './ControlledPhysiology.jsx';")
text=text.replace('<TrainingEvidence/><PhysiologyEvidence/><div','<TrainingEvidence/><ControlledPhysiology/><details className="panel"><summary>Earlier pooled analysis — game / label confounding</summary><PhysiologyEvidence/></details><div')
path.write_text(text,encoding='utf-8')
path=root/'PROJECT_STATUS.md';text=path.read_text(encoding='utf-8');marker='<!-- further-experiment-active -->'
if marker not in text:path.write_text(marker+'\nFurther FusDenseNet training and fixed-context real physiology evaluation are in progress. The deployed 52.55% checkpoint is retained until a new candidate is evaluated. See reports/FURTHER_EXPERIMENT_PLAN.md.\n'+marker+'\n\n'+text,encoding='utf-8')
