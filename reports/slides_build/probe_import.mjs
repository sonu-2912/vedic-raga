import fs from 'node:fs/promises';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
const p=await PresentationFile.importPptx(await FileBlob.load('C:/Users/bindu/OneDrive/Desktop/VEDIC RAGA/output/RagaMoodSync_Research_Real_Data_Final.pptx'));
console.log('slides',p.slides.items.length,'charts',p.slides.items[8].charts.items.length,'tables',p.slides.items[9].tables.items.length);
console.log('masters',p.masters.items.length);
await fs.writeFile('C:/Users/bindu/OneDrive/Desktop/VEDIC RAGA/reports/slides_build/imported-source.inspect.ndjson',(await p.inspect({kind:'layout,textbox,chart,table',maxChars:150000})).ndjson);
console.log(await(await p.slides.items[8].export({format:'layout'})).text());
