import React from 'react';
import {createRoot} from 'react-dom/client';
const research = new URLSearchParams(location.search).get('view') === 'research';
async function boot(){
 let module;
 if(research){module=await import('./research/ResearchStudio.jsx')}
 else{module=await import('./ListenerApp.jsx')}
 createRoot(document.getElementById('root')).render(React.createElement(module.default));
}
boot();
