import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';

const root=path.resolve(import.meta.dirname,'../..');
const skill='C:/Users/bindu/.codex/plugins/cache/openai-primary-runtime/presentations/26.1026.10807/skills/presentations';
const runtimePython='C:/Users/bindu/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
process.env.RUNTIME_NODE_MODULES='C:/Users/bindu/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const read=async name=>JSON.parse(await fs.readFile(path.join(root,'reports',name),'utf8'));
const smoke=await read('synthetic_emotion_smoke.json');
const signals=await read('sample_signal_results.json');
const e2e=await read('end_to_end.json');
const real=await read('emotion_evaluation.json');
const history=await read('training_history.json');
const verified=await read('verification_summary.json');
const macro=real.classification_report['macro avg'];
const comparison=await read('model_comparison.json');
const training=verified.training;
const parameters=verified.real_fer2013_metrics.parameters;
const hog=await read('hog_baseline.json');
const multimodal=await read('multimodal_extension_evaluation.json');
const gamer=await read('gamer_evaluation.json');
const primary=gamer.tasks.anger_vs_sadness;
const five=gamer.tasks.five_class;
const improvement=await read('emotion_training_comparison.json');
const paper='https://doi.org/10.1007/s41870-025-02531-y';
const refs={paper,fer:'https://doi.org/10.1007/978-3-642-42051-1_16',dense:'https://doi.org/10.1109/CVPR.2017.243',ncf:'https://doi.org/10.1145/3038912.3052569',mathur:'https://doi.org/10.3389/fpsyg.2015.00513',valla:'https://doi.org/10.3389/fpsyg.2017.02115',deap:'https://doi.org/10.1109/T-AFFC.2011.15',wesad:'https://doi.org/10.1145/3242969.3242985',saraga:'https://doi.org/10.18061/emr.v16i1.7641',mne:'https://doi.org/10.3389/fnins.2013.00267',nk:'https://doi.org/10.3758/s13428-020-01516-y',patent:'https://patents.google.com/patent/US20230153350A1/en',repo:'https://github.com/alronova/raag-ratnakar'};
const p=Presentation.create({slideSize:{width:1280,height:720}});
const C={bg:'#F6F5EF',ink:'#223E34',muted:'#67786B',accent:'#A47546',soft:'#E7EDDF',line:'#C9D4C3'};
function text(s,value,x,y,w,h,size=26,bold=false,color=C.ink){const a=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0,style:'solid'}});a.text=value;a.text.style={typeface:'Arial',fontSize:size,bold,color};a.text.insets=0;return a;}
function slide(title,subtitle='',sources=[]){const s=p.slides.add();s.background.fill=C.bg;const n=p.slides.items.length;text(s,title,64,49,1140,65,40,true);if(subtitle)text(s,subtitle,64,120,1150,48,21,false,C.muted);text(s,String(n).padStart(2,'0')+' / 13',1140,675,80,20,13,false,C.muted);if(sources.length){text(s,sources.map(x=>x[0]).join('  ·  '),64,654,1050,44,12,false,C.muted);s.speakerNotes.text=sources.map(x=>x[0]+': '+x[1]).join('\n');}return s;}
function paragraphs(s,items,x=64,y=190,w=1100,size=26,gap=110){items.forEach((t,i)=>text(s,t,x,y+i*gap,w,gap-18,size));}
function note(s,v){s.speakerNotes.text=String(s.speakerNotes.text||'')+'\n'+v;}
async function image(s,name,x,y,w,h){const bytes=await fs.readFile(path.join(root,'reports',name));s.images.add({blob:bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),contentType:name.endsWith('.png')?'image/png':'image/jpeg',alt:name,fit:'contain',position:{left:x,top:y,width:w,height:h}});}
function table(s,values,x,y,w,h,widths,size=20){const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,columnWidths:widths,values});for(let r=0;r<values.length;r++){t.rows[r].height=h/values.length;for(let c=0;c<values[0].length;c++){const cell=t.getCell(r,c);cell.fill=r===0?C.ink:(r%2?C.bg:C.soft);cell.text.style={typeface:'Arial',fontSize:size,color:r===0?'#FFFFFF':C.ink,bold:r===0};}}t.borders.assign({fill:C.line,width:.6,style:'solid'});return t;}
function chartFont(c){applyPresentationChartFont(c,{fontFamily:'Arial'});return c;}

// 1. Explicitly editable group placeholders.
{
 const s=slide('RagaMoodSync','Facial expression, physiological context and Hindustani raga research',[['Pingle & Ragha, 2025',paper]]);
 text(s,'Music, expression\nand personal listening',64,210,900,170,58,true);
 text(s,'Group: [Group name]    Members: [Names / IDs]\nGuide: [Guide name]    Institution: [Institution]',64,445,1120,100,25);
 text(s,'Paper-informed implementation · Windows CPU · Research prototype',64,585,1110,40,22,false,C.accent);
 note(s,'Editable group placeholders intentionally retained. No group identities were supplied. Project does not claim exact reproduction or clinical validation.');
}
// 2.
{
 const s=slide('Motivation and significance','Music discovery can reflect a listener’s expression, cultural context and feedback.',[['Mathur et al., 2015',refs.mathur],['Valla et al., 2017',refs.valla]]);
 paragraphs(s,['Facial expressions offer a convenient, uncertain input that a user can correct.','Hindustani metadata adds cultural context beyond generic genre labels.','Persistent feedback lets recommendations adapt to individual experience.'],64,190,1080,29,115);
 text(s,'Research objective: build and test the full software loop. Therapeutic efficacy remains unestablished.',64,570,1100,60,24,true,C.accent);
}
// 3.
{
 const s=slide('Literature review','The selected paper defines the architecture. Adjacent work supplies context.',[['Pingle & Ragha, 2025',paper],['Goodfellow et al., 2013',refs.fer],['Huang et al., 2017',refs.dense],['He et al., 2017',refs.ncf],['Mathur et al., 2015',refs.mathur]]);
 table(s,[['Research','Contribution','Project implication'],['RagaMoodSync (2025)','FusDenseNet and FDCF','Implement mechanisms, disclose assumptions'],['FER challenge (2013)','Facial-expression benchmark','Keep held-out classification evaluation'],['DenseNet (2017)','Dense feature reuse','Use concatenated earlier features'],['Neural CF (2017)','Learned neural matching','Learn user/music interaction representations'],['Raga emotion study (2015)','Listener-rated musical structure','Do not equate a raga name with a universal outcome']],64,188,1152,402,[250,410,492],22);
 text(s,'FER2013 training and expanded test evaluation completed. Real listener trials remain pending.',64,610,1130,30,21,false,C.accent);
}
//4.
{
 const s=slide('Prior art and preliminary IP search','The broad concept of facial-emotion music recommendation is already disclosed.',[['US20230153350A1',refs.patent],['RagaMoodSync',paper],['Raag Ratnakar',refs.repo]]);
 text(s,'US20230153350A1',64,200,1100,60,39,true,C.accent);
 paragraphs(s,['“Music recommendation system by facial emotion using deep learning”\nPublication: 18 May 2023. Description includes audio preprocessing and attention.','Search scope: targeted web/Google Patents review. Official claims, families, prosecution and jurisdictional coverage remain to be assessed.','Rights: the supplied Raag Ratnakar archive contains no license file. Preserve attribution and resolve permission before redistribution.'],64,280,1110,25,112);
 note(s,'Preliminary technical review, not patentability or freedom-to-operate clearance. Dates are bibliographic page information. No novelty or legal-status conclusion is claimed.');
}
//5.
{
 const s=slide('Relevance, applications and end users','A local research studio for culturally informed music exploration.');
 table(s,[['End user','Use','Boundary'],['Listeners','Explore vocal/instrumental raga links','User can correct inferred mood'],['Students and educators','Inspect source metadata and missing fields','Associations require expert review'],['AI / MIR researchers','Train models and evaluate feedback loops','Synthetic tests do not show real-world quality'],['Future study teams','Investigate consented affect/music responses','Clinical use requires separate evidence']],64,205,1152,340,[270,450,432],23);
 text(s,'Windows CPU execution makes the prototype usable without a GPU or physical sensors.',64,585,1110,54,27,true,C.accent);
}
//6.
{
 const s=slide('Proposed solution and innovation','An executable integration, with mechanisms and evidence kept traceable.',[['Paper architecture',paper],['Initial knowledge source',refs.repo]]);
 paragraphs(s,['Operational loop: image or manual mood, raga ranking, links, feedback and listening history.','Paper-informed models: repeated fusion-dense stages, cross-attention, multi-pooling and a trainable FDCF focus network.','Engineering contribution: field provenance, explicit unknown metadata, optional signal inputs and reproducible CPU experiments.'],64,190,1110,28,120);
 text(s,'Default: locally trained FusDenseNet v2. Supplied Navrasa remains a separately labeled baseline.',64,575,1110,62,23,true,C.accent);
 note(s,'No scientific novelty or measured recommendation benefit is asserted. Native nine-Rasa baseline probabilities map heuristically to seven emotion labels.');
}
//7. Native editable architecture diagram, as required.
{
 const s=slide('System architecture','Confirmed feelings, listening goals, weather and recorded body context shape the ranking.',[['Paper core: Figs 1–4 and Eqs 3–15',paper],['Weather extension','https://open-meteo.com/en/docs']]);
 const node=(v,x,y,w=205,h=96)=>{const a=s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill:C.soft,line:{fill:C.line,width:1,style:'solid'}});a.text=v;a.text.style={typeface:'Arial',fontSize:22,color:C.ink};return a;};
 const a=node('Webcam /\nimage upload',64,210),b=node('FusDenseNet\nUncertain outputs\nare withheld',310,210),c=node('Confirmed feeling\nListening goal',556,210),d=node('IKS + preferences\nOptional trained FDCF',802,210,285);
 const ee=node('EEG / HRV\nCity / manual weather\nOptional context',64,420,240),ff=node('SQLite\nFeedback and history',802,420,285),rr=node('Context-aware ranking\nRaga details + links',443,420,295);
 for(const[from,to]of[[a,b],[b,c],[c,d]])s.shapes.connect(from,to,{kind:'straight',fromSide:'right',toSide:'left',line:{fill:C.ink,width:2,style:'solid'},tail:{type:'triangle'}});
 s.shapes.connect(d,rr,{fromSide:'bottom',toSide:'top',kind:'elbow',line:{fill:C.ink,width:2,style:'solid'},tail:{type:'triangle'}});
 s.shapes.connect(rr,ff,{fromSide:'right',toSide:'left',kind:'straight',line:{fill:C.ink,width:2,style:'solid'},tail:{type:'triangle'}});
 s.shapes.connect(ff,d,{fromSide:'top',toSide:'bottom',kind:'straight',line:{fill:C.accent,width:2,style:'dashed'},tail:{type:'triangle'}});
 s.shapes.connect(ee,rr,{fromSide:'right',toSide:'left',kind:'straight',line:{fill:C.accent,width:2,style:'dashed'},tail:{type:'triangle'}});
 text(s,'FusDenseNet v2 retains dense fusion, cross-attention and spatial/global pooling.\nWeather/body rules adjust listening context; they do not identify the person’s mood.',64,565,1150,65,22);
}
//8.
{
 const s=slide('Dataset sources and methodology','Facial benchmark, optional recordings and live/manual weather have different evidence roles.',[['GamER','https://doi.org/10.5281/zenodo.8191877'],['FER2013',refs.fer],['DEAP',refs.deap],['WESAD',refs.wesad],['Saraga',refs.saraga],['Open-Meteo','https://open-meteo.com/en/terms']]);
 table(s,[['Resource','Role','Access / current evidence'],['Raag Ratnakar','Metadata and Navrasa weights','72 ragas, 528 entries, 516 unique URLs'],['FER2013','Real facial CPU training / test','Version 1, 63.3 MB ZIP. ODbL / DbCL advertised'],['DEAP / WESAD','Physiological affect benchmarks','Access review. Not raga-response data'],['GamER','Real paired EEG + ECG evaluation','Official CC BY 4.0 release; participant-held-out'],['Saraga / CompMusic','Future features and metadata','Saraga data: CC BY-NC. Check each source'],['IndRaga','Future precomputed audio features','Primary DOI record unavailable in review']],64,188,1152,410,[240,390,522],20);
 text(s,'Weather: Open-Meteo or manual entry. Model-based city data; experimental Ritu preference.',64,620,1120,30,21,true,C.accent);
 note(s,'Source links: https://www.kaggle.com/datasets/msambare/fer2013 ; https://www.eecs.qmul.ac.uk/mmv/datasets/deap/ ; https://archive.ics.uci.edu/dataset/465/wesad ; https://physionet.org/ ; https://mtg.github.io/saraga/ ; https://doi.org/10.5281/zenodo.19978345 ; https://compmusic.upf.edu/ . IndRaga primary metadata/files/license were not verified.');
}
//9. Native chart plus actual scientific confusion-matrix raster.
{
 const s=slide('FusDenseNet implementation and results',`CPU: ${history.filter(r=>!r.phase).length} full-network + ${history.filter(r=>r.phase).length} head-only recorded epochs · selected index ${real.checkpoint_epoch}`,[['Figs 2–3, Eqs 3–9',paper],['Actual project evidence','reports/emotion_evaluation.json'],['HOG method: Dalal & Triggs, 2005','https://doi.org/10.1109/CVPR.2005.177']]);
 text(s,`${parameters.toLocaleString('en-US')} parameters · spatial + global pooling\n3 fusion stages, cross-attention, dilation\n${training.train_count.toLocaleString()} train / ${training.val_count.toLocaleString()} validation`,64,185,640,100,23);
 chartFont(s.charts.add('line',{position:{left:64,top:310,width:610,height:280},title:'FER2013 curves; final phase trains MLP only',categories:history.map(x=>String(x.epoch)),series:[{name:'Train accuracy',values:history.map(x=>Number(x.accuracy.toFixed(6))),line:{fill:C.ink,width:2,style:'solid'}},{name:'Validation accuracy',values:history.map(x=>Number(x.val_accuracy.toFixed(6))),line:{fill:C.accent,width:2,style:'solid'}}],hasLegend:true,xAxis:{title:'Epoch',textStyle:{typeface:'Arial',fontSize:13}},yAxis:{textStyle:{typeface:'Arial',fontSize:13},numberFormatCode:'0%',majorGridlines:{fill:C.line,width:1,style:'solid'}}}));
 await image(s,'emotion_evaluation.png',750,170,420,350);
 text(s,`Held-out n=${real.n}, accuracy ${(100*real.accuracy).toFixed(2)}%\nMacro precision ${macro.precision.toFixed(4)}, recall ${macro.recall.toFixed(4)}\nMacro F1 ${macro['f1-score'].toFixed(4)}`,700,530,520,90,21,true,C.accent);
 text(s,`Same-test accuracy: pilot ${(100*comparison.baseline.accuracy).toFixed(2)}% · HOG/SVM ${(100*hog.accuracy).toFixed(2)}% · revised ${(100*real.accuracy).toFixed(2)}%.`,64,620,1140,30,22,true,C.accent);
 note(s,`Actual project training from random initialization, not supplied Navrasa or paper weights. Train ${training.train_count}, validation ${training.val_count}, test ${real.n}, seed 42. Exact grayscale-pixel duplicates removed. Combined Kaggle test folder includes examples previously used in the old pilot evaluation, not a fresh dataset or original FER PrivateTest-only/identity-disjoint protocol. Best checkpoint epoch ${real.checkpoint_epoch}, SHA-256 ${real.checkpoint_sha256}. Architecture v2 adds normalized inputs/fusion and spatial 2x2 pooling; all dimensions are project assumptions. AdamW/cosine, class-weighted smoothed CE + 0.05 reconstruction MSE. Accuracy ${real.accuracy}, macro precision ${macro.precision}, recall ${macro.recall}, F1 ${macro['f1-score']}. Baseline reevaluated on exactly the same retained test faces: accuracy ${comparison.baseline.accuracy}, macro F1 ${comparison.baseline.macro_f1}. The selected candidate may include classifier-only refinement on frozen features, with training-fitted normalization folded into the existing MLP. Epoch indices combine full-network and head phases; they must not be read as equal full-image training budgets. This is not a controlled architecture ablation. Paper Table 2 accuracy 99.72% and Table 5 accuracy 99.82% are inconsistent and not reproduced.`);
}
//10. Actual source-labelled physiological benchmark, native editable table.
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
//12.
{
 const s=slide('Future work and publication possibilities','Publication depends on sound data, rights and evaluation.');
 paragraphs(s,['Train equal-budget attention / pooling ablations with multiple seeds, stronger split controls and calibrated uncertainty.','Collect consented listening feedback and expert-reviewed raga annotations. Evaluate chronological ranking and IKS ablation.','Extend the real physiology baseline with independent signal-quality review, confirmed annotation alignment and external participant validation.','A reproducibility or systems paper is plausible after these experiments. Clinical and patent claims require separate substantiation.'],64,185,1110,27,107);
 note(s,'No publication acceptance, novelty, clinical efficacy or patentability is promised. Resolve third-party permissions before releasing catalog or checkpoint assets.');
}
//13. Complete references for claims in this deck, in editable text.
{
 const s=slide('Complete references','Academic, data and prior-art sources cited throughout the presentation');
 const left=[
 '1. Pingle, Y. P. & Ragha, L. K. (2025). RagaMoodSync: a deep learning framework for Hindustani Raga recommendations driven by facial emotions. International Journal of Information Technology, 17, 2765–2778. DOI: 10.1007/s41870-025-02531-y.',
 '2. Goodfellow, I. et al. (2013). Challenges in Representation Learning: A report on three machine learning contests. DOI: 10.1007/978-3-642-42051-1_16.',
 '3. Huang, G., Liu, Z., van der Maaten, L. & Weinberger, K. Q. (2017). Densely Connected Convolutional Networks. CVPR. DOI: 10.1109/CVPR.2017.243.',
 '4. He, X. et al. (2017). Neural Collaborative Filtering. WWW. DOI: 10.1145/3038912.3052569.',
 '5. Mathur, A. et al. (2015). Emotional responses to Hindustani raga music: the role of musical structure. Frontiers in Psychology, 6, 513. DOI: 10.3389/fpsyg.2015.00513.',
 '6. Valla, J. M. et al. (2017). Music and Emotion—A Case for North Indian Classical Music. Frontiers in Psychology, 8, 2115. DOI: 10.3389/fpsyg.2017.02115.',
 '7. Koelstra, S. et al. (2012). DEAP: A Database for Emotion Analysis Using Physiological Signals. IEEE TAC. DOI: 10.1109/T-AFFC.2011.15.'
  ];
 left.push('16. Papadopoulos, A., Episkopou, C., Malegiannaki, A., Kyritsis, K., Sarafis, I., Papaefstathiou, I. & Delopoulos, A. (2023). GamER: Gameplay Physiological Signal Dataset for Emotion Recognition. Zenodo v1. DOI: 10.5281/zenodo.8191877. CC BY 4.0.');
 const right=[
 '8. Schmidt, P. et al. (2018). Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection. ICMI. DOI: 10.1145/3242969.3242985.',
 '9. Srinivasamurthy, A. et al. (2021). Saraga: Open Datasets for Research on Indian Art Music. Empirical Musicology Review, 16, 85–98. DOI: 10.18061/emr.v16i1.7641.',
 '10. Gramfort, A. et al. (2013). MEG and EEG data analysis with MNE-Python. DOI: 10.3389/fnins.2013.00267.',
 '11. Makowski, D. et al. (2021). NeuroKit2: A Python toolbox for neurophysiological signal processing. DOI: 10.3758/s13428-020-01516-y.',
 '12. US20230153350A1 (2023). Music recommendation system by facial emotion using deep learning. patents.google.com/patent/US20230153350A1/en.',
 '13. alronova. Raag Ratnakar. github.com/alronova/raag-ratnakar. Supplied ZIP, license unresolved. NASA portrait provenance: scikit-image.org/docs/stable/api/skimage.data.html.',
 '14. Dalal, N. & Triggs, B. (2005). Histograms of Oriented Gradients for Human Detection. CVPR, pp. 886–893. DOI: 10.1109/CVPR.2005.177. HOG baseline here uses project-selected FER settings.',
 '15. Open-Meteo. Forecast / Geocoding API documentation and Terms. open-meteo.com/en/docs; open-meteo.com/en/docs/geocoding-api; open-meteo.com/en/terms.',
 'Data portals: Kaggle msambare/fer2013; eecs.qmul.ac.uk/mmv/datasets/deap; archive.ics.uci.edu/dataset/465/wesad; physionet.org; mtg.github.io/saraga; compmusic.upf.edu. IndRaga DOI: 10.5281/zenodo.19978345 (primary access unresolved).'
 ];
 text(s,left.join('\n'),64,180,553,480,16);
 text(s,right.join('\n'),662,180,554,480,16);
 note(s,Object.entries(refs).map(([name,url])=>name+': '+url).join('\n')+'\nFull titles, access notes and license status are in LITERATURE_REVIEW.md, DATASETS.md and PRIOR_ART_AND_IP.md. Source access review: 2 November 2026.');
}

if(p.slides.items.length!==13)throw Error('Expected exactly 13 slides');
const build=path.join(root,'reports/slides_build');await fs.mkdir(build,{recursive:true});
for(let i=0;i<p.slides.items.length;i++){
 const s=p.slides.items[i];const b=await p.export({slide:s,format:'png',scale:1});await fs.writeFile(path.join(build,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await b.arrayBuffer()));
 const layout=await s.export({format:'layout'});await fs.writeFile(path.join(build,`slide-${String(i+1).padStart(2,'0')}.layout.json`),await layout.text());
}
const candidate=path.join(build,'candidate.pptx');await(await PresentationFile.exportPptx(p)).save(candidate);
await fs.writeFile(path.join(build,'deck.inspect.ndjson'),(await p.inspect({kind:'slide,textbox,chart,table,notes',maxChars:150000})).ndjson);
const finalPath=path.join(root,'output/RagaMoodSync_Research_Real_Data_Final.pptx');
const result=await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath,pythonExecutable:runtimePython,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[3,5,8,10].flatMap(n=>['--require-native-table-slide',String(n)])],explicitTotalSlideCount:13,requiredNativeTableOwnerSlides:[3,5,8,10],requiredNativeChartOwnerSlides:[9],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,receiptPath:path.join(build,'validation-real-data-final.json')});
console.log(JSON.stringify(result));

