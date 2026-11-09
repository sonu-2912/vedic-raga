import {FileBlob,PresentationFile} from '@oai/artifact-tool';
import path from 'node:path';
const p=await PresentationFile.importPptx(await FileBlob.load(path.resolve('output/RagaMoodSync_Research_Real_Data_Final.pptx')));
const slide=p.slides.items[8],c=slide.charts.items[0];
for(const [name,o] of [['collection',slide.charts],['chart',c],['slide',slide]]) console.log(name,Object.getOwnPropertyNames(Object.getPrototypeOf(o)).join(','));
