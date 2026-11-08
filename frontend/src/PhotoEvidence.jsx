import React from 'react';
export default function PhotoEvidence({prediction:p}){
 if(!p?.probabilities)return null;
 return <details className="tradition-details"><summary>See the model’s actual prediction</summary><p>Raw expression guess: <strong>{p.emotion}</strong> ({(100*p.confidence).toFixed(1)}% model probability). This may be wrong; it is not a measurement of your internal mood.</p><ul>{Object.entries(p.probabilities).sort((a,b)=>b[1]-a[1]).map(([label,value])=><li key={label}>{label}: {(100*value).toFixed(1)}%</li>)}</ul>{p.held_out_evaluation&&<p>Full seven-class test accuracy: {(100*p.held_out_evaluation.accuracy).toFixed(2)}%. Macro F1: {p.held_out_evaluation.macro_f1.toFixed(4)}. All test classes are included, including wrong predictions.</p>}</details>
}
