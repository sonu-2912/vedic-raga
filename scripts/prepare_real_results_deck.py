"""Prepare the existing 13-slide design with real-data evidence bindings."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
source=(root/'reports/slides_build/build_deck.mjs').read_text(encoding='utf-8')
source=source.replace("const bodyDemo=multimodal.scenarios.all_contexts.context.physiology.features;","const gamer=await read('gamer_evaluation.json');\nconst primary=gamer.tasks.anger_vs_sadness;\nconst five=gamer.tasks.five_class;\nconst improvement=await read('emotion_training_comparison.json');")
source=source.replace('A facial emotion-driven Hindustani raga research application','Facial expression, physiological context and Hindustani raga research')
source=source.replace('`CPU training: ${history.length} epochs · selected epoch ${real.checkpoint_epoch} by validation macro F1`',"`CPU: ${history.filter(r=>!r.phase).length} full-network + ${history.filter(r=>r.phase).length} head-only recorded epochs · selected index ${real.checkpoint_epoch}`")
source=source.replace("title:'Actual FER2013 training curves'","title:'FER2013 curves; final phase trains MLP only'")
source=source.replace("'Actual project training from random initialization", "'Actual project training from random initialization")
source=source.replace("This is not a controlled architecture ablation. Paper Table 2", "The selected candidate may include classifier-only refinement on frozen features, with training-fitted normalization folded into the existing MLP. Epoch indices combine full-network and head phases; they must not be read as equal full-image training budgets. This is not a controlled architecture ablation. Paper Table 2")
source=source.replace("['PhysioNet','Future ECG/PPG resource','Per-dataset access and license']","['GamER','Real paired EEG + ECG evaluation','Official CC BY 4.0 release; participant-held-out']")
source=source.replace("[['FER2013',refs.fer],['DEAP',refs.deap]","[['GamER','https://doi.org/10.5281/zenodo.8191877'],['FER2013',refs.fer],['DEAP',refs.deap]")
start=source.index('//10.');end=source.index('//12.')
source=source[:start]+r'''//10. Actual source-labelled physiological benchmark, native editable table.
{
 const s=slide('EEG / HRV, weather, IKS and FDCF',`Real GamER: 2-class ${primary.n_intervals} intervals / ${primary.n_subjects} people; 5-class ${five.n_intervals} / ${five.n_subjects}. Separate from FER2013 and raga outcomes.`,[['GamER: Papadopoulos et al., 2023','https://doi.org/10.5281/zenodo.8191877'],['FDCF Eqs 10–15',paper],['NeuroKit2',refs.nk]]);
 text(s,'FDCF + IKS',64,190,535,45,30,true);
 text(s,'User/music/emotion embeddings\nFocus, dynamic preferences and MLP\nBhava, Prahara, Ritu: source-supplied\nMissing raga fields remain unknown',64,247,535,160,24);
 text(s,'What was actually evaluated',64,450,550,45,27,true);
 text(s,'EEG vs cardiac vs feature fusion\nHeld-out people; short RR windows\nNo artificial pairing with FER2013 faces\nOffset sign assumed; sensitivity tested\nWeather context remains experimental',64,500,550,140,22);
 const rows=[['Input','2-class acc.','2-class F1','5-class F1'],...['EEG','Cardiac','Fusion','Majority','Game ID control'].map(name=>[name==='Game ID control'?'Game only':name,(100*primary.results[name].accuracy).toFixed(2)+'%',primary.results[name].macro_f1.toFixed(3),five.results[name].macro_f1.toFixed(3)])];
 table(s,rows,650,205,566,300,[126,150,145,145],19);
 text(s,`5-class accuracy: ${(100*five.results.Fusion.accuracy).toFixed(2)}% < majority ${(100*five.results.Majority.accuracy).toFixed(2)}%.\nFusion − EEG 2-class F1: ${primary.fusion_minus_eeg_macro_f1.toFixed(3)}\n95% interval: [${primary.subject_bootstrap_95pct_delta_interval.map(x=>x.toFixed(3)).join(', ')}]; includes zero.\nGame ID alone: ${(100*primary.results['Game ID control'].accuracy).toFixed(0)}% → task confounding.`,650,530,566,105,21,true,C.accent);
 note(s,JSON.stringify({protocol:gamer.protocol,primary:primary.results,five_class:five.results,delta_interval:primary.subject_bootstrap_95pct_delta_interval})+' Primary labels Angry/Sad. Secondary labels Angry/Sad/Bored/Stressed/Happy. StandardScaler fitted inside each training-subject fold; fixed logistic regression C=1 and balanced loss. Cardiac features use real ECG-derived RR intervals, often shorter than 60 seconds, so they are exploratory ultra-short statistics, not clinical HRV. No disgust, face, weather or raga response is present. Both offset-sign conventions are reported separately; no favorable sign is selected after testing. Negative deltas are regressions. Original source annotations are psychologist assessments, not universal emotion ground truth.');
}
//11.
{
 const s=slide('Application, experiments and limitations','Unfiltered facial metrics and real-recording fusion evidence are visible in the app.',[['Project tests','reports/pytest.xml'],['Actual physiology','reports/gamer_evaluation.json'],['Actual emotion comparison','reports/colab_comparison.json']]);
 await image(s,'research_real_physiology.jpg',64,185,655,440);
 text(s,`${verified.passed} / ${verified.tests} automated tests pass`,765,195,440,65,30,true);
 const delta=improvement.metrics;
 text(s,`New selected model vs prior 50.13% model:\nAccuracy: ${(100*delta.accuracy.baseline).toFixed(2)}% → ${(100*delta.accuracy.colab).toFixed(2)}%\nMacro F1: ${delta.macro_f1.baseline.toFixed(4)} → ${delta.macro_f1.colab.toFixed(4)}\nMacro precision: ${delta.macro_precision.baseline.toFixed(4)} → ${delta.macro_precision.colab.toFixed(4)}\nMacro recall: ${delta.macro_recall.baseline.toFixed(4)} → ${delta.macro_recall.colab.toFixed(4)}\nDisgust recall: ${(100*delta.disgust_recall.baseline).toFixed(2)}% → ${(100*delta.disgust_recall.colab).toFixed(2)}%\nAll seven labels and retained test faces.`,765,270,450,220,21);
 text(s,'No validated webcam-domain accuracy.\nGamER game / label confounding.\nNo face–physiology joint benchmark.\nNo raga or weather outcome evidence.',765,530,450,110,20,true,C.accent);
 note(s,'Screenshot shows the actual participant-held-out physiology report. The new FER experiment uses validation-only checkpoint selection; the historically inspected combined test folder is not fresh or identity-disjoint. All errors remain included. Deployment is gated on full-test accuracy and macro F1 increasing without lower disgust recall. If that gate fails, the app retains its old classifier and reports the new experiment separately. No sample/simulated physiological result is used for the real-data metrics.');
}
''' + source[end:]
source=source.replace("'Validate physiology with signal-quality checks and participant-disjoint data before learning multimodal fusion.'","'Extend the real physiology baseline with independent signal-quality review, confirmed annotation alignment and external participant validation.'")
source=source.replace("'7. Koelstra", "'7. Koelstra",1)
# Put the additional dataset reference into the left column, which has room.
needle="const right=["
location=source.index(needle)
before=source[:location];after=source[location:]
last=before.rfind('];')
before=before[:last]+before[last:].replace('];'," ];\n left.push('16. Papadopoulos, A., Episkopou, C., Malegiannaki, A., Kyritsis, K., Sarafis, I., Papaefstathiou, I. & Delopoulos, A. (2023). GamER: Gameplay Physiological Signal Dataset for Emotion Recognition. Zenodo v1. DOI: 10.5281/zenodo.8191877. CC BY 4.0.');",1)
source=before+after
source=source.replace('output/RagaMoodSync_Research_Multimodal.pptx','output/RagaMoodSync_Research_Real_Data_Final.pptx').replace("validation-multimodal.json","validation-real-data-final.json")
(root/'reports/slides_build/build_real_deck.mjs').write_text(source,encoding='utf-8')
print('Prepared real-data revision of the existing 13-slide design')
