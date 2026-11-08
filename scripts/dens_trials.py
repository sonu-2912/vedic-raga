"""Acquire a predeclared non-game EEG/ECG cohort with exact 60s signal ranges.

Metadata and full EEG headers are checked against official annex MD5 references.
Partial FDT ranges are TLS/Content-Range/length checked and locally SHA256-hashed;
they are not misrepresented as full-FDT-checksum verified downloads.
"""
import hashlib,io,json,re,shutil,time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
import pandas as pd
import requests
from scipy.io import loadmat
ROOT=Path(__file__).resolve().parents[1];FOLDER=ROOT/'data/private/dens';FOLDER.mkdir(exist_ok=True,parents=True)
RAW='https://raw.githubusercontent.com/OpenNeuroDatasets/ds003751/master/'
S3='https://s3.amazonaws.com/openneuro.org/ds003751/'
PROTOCOL={'dataset':'DENS ds003751; official OpenNeuro data','participants':'First 12 lexicographically ordered released participant folders; no score-based subject selection','task':'Participant-rated valence: low <=4, high >=6; intermediate ratings excluded','window':'Centered exactly 60s inside stimulus; no padding or cross-trial extension','timing':'EEGLAB event latencies are 1-based samples, checked against source ISO event time differences and header sampling rate','evaluation':'4x4 crossed participant/stimulus held-out folds; training excludes every held-out participant AND every held-out video ID','models':'Fixed fold-fitted StandardScaler + balanced LogisticRegression C=1; EEG, cardiac, fusion and training-majority baseline','teacher':None,'game_data':False,'licence':'dataset_description says CC0; README/LICENSE says ODbL/DbCL. Preserve both and use more protective attribution/share-alike conditions for derived data','access':'Public official repository/bucket, no credentials or signed EULA','partial_signal_integrity':'Exact HTTP byte range and local SHA256; no full-FDT MD5 verification claim','seed':94,'limitations':['Small 12-participant subset','Physiological and annotation artifacts remain possible','60s RR features are not five-minute clinical HRV','No facial FER labels, disgust target or raga-response outcomes']}

def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
def get(url):
    r=requests.get(url,timeout=(20,90));r.raise_for_status();return r.content
def annex(pointer):
    m=re.search(r'MD5E-s(\d+)--([0-9a-f]{32})(?:\.[A-Za-z0-9]+)?',pointer.decode());assert m,'Not an official MD5 annex pointer';return int(m[1]),m[2]
def checked(url,size,md5,path):
    path=Path(path)
    if path.exists():data=path.read_bytes()
    else:data=get(url)
    if len(data)!=size or hashlib.md5(data).hexdigest()!=md5:raise ValueError('Full header checksum mismatch')
    path.write_bytes(data);return data

def main():
    protocol=ROOT/'reports/dens_protocol.json'
    if protocol.exists() and json.loads(protocol.read_text())!=PROTOCOL:raise ValueError('Protocol changed')
    write(protocol,PROTOCOL)
    tree=json.loads((ROOT/'reports/dens_access/tree.json').read_text())['tree'];paths=[x['path'] for x in tree if x['type']=='blob']
    subjects=sorted({p.split('/')[0] for p in paths if p.startswith('sub-')})[:12]
    assert shutil.disk_usage(ROOT).free>20_000_000_000
    # Metadata for independent participants can be acquired concurrently. The
    # fixed cohort and integrity checks are unchanged by transfer order.
    def prepare(subject):
        directory=FOLDER/subject;directory.mkdir(exist_ok=True)
        for ending in ['.tsv','.set','.fdt']:
            candidates=[p for p in paths if p.startswith(subject+('/beh/' if ending=='.tsv' else '/eeg/')) and p.endswith(ending)]
            p=candidates[0];dest=directory/(Path(p).name+('' if ending=='.tsv' else '.annex.txt'))
            if not dest.exists():dest.write_bytes(get(RAW+p))
            if ending=='.set':
                size,md5=annex(dest.read_bytes());checked(S3+p,size,md5,directory/Path(p).name)
        print('Prepared metadata '+subject,flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(prepare,subjects))
    trials=[];audit=[];excluded=[]
    for subject in subjects:
        directory=FOLDER/subject;directory.mkdir(exist_ok=True)
        beh=next(p for p in paths if p.startswith(subject+'/beh/') and p.endswith('.tsv'))
        header=next(p for p in paths if p.startswith(subject+'/eeg/') and p.endswith('.set'))
        fdt=next(p for p in paths if p.startswith(subject+'/eeg/') and p.endswith('.fdt'))
        bpath=directory/Path(beh).name
        if not bpath.exists():bpath.write_bytes(get(RAW+beh))
        ratings=pd.read_csv(bpath,sep='\t')
        pointers={}
        for p in [header,fdt]:
            ptr=directory/(Path(p).name+'.annex.txt')
            if not ptr.exists():ptr.write_bytes(get(RAW+p))
            pointers[p]=annex(ptr.read_bytes())
        hs,hm=pointers[header];hpath=directory/Path(header).name;h=checked(S3+header,hs,hm,hpath)
        eeg=loadmat(io.BytesIO(h),simplify_cells=True)['EEG'];fs=float(eeg['srate']);channels=int(eeg['nbchan']);points=int(eeg['pnts']);size,md5=pointers[fdt]
        if points*channels*4!=size or fs<80:
            excluded.append({'subject':subject,'reason':'Official header dimensions do not match referenced float32 FDT length','expected_bytes':points*channels*4,'reference_bytes':size})
            print(json.dumps(excluded[-1]),flush=True)
            continue
        labels=[str(x['labels']) for x in eeg['chanlocs']];ecg=[i for i,n in enumerate(labels) if n.upper() in ['ECG','EKG']];assert len(ecg)==1
        eeg_indices=[i for i,n in enumerate(labels) if n.startswith('E') and n[1:].isdigit()]
        events=eeg['event'];events=[events] if isinstance(events,dict) else list(events)
        times=[]
        for event in events:
            try:times.append((float(event['latency']),datetime.fromisoformat(str(event['begintime'])).timestamp()))
            except (ValueError,TypeError,KeyError):pass
        assert len(times)>=2
        error=max(abs((a-times[0][0])/fs-(b-times[0][1])) for a,b in times)
        # EEGLAB boundary events mark removed recording pauses. A global wall
        # clock offset can change across them; validate local trial timing and
        # reject every trial crossing a boundary rather than guess an offset.
        audit.append({'subject':subject,'header_md5':hm,'fdt_reference_md5':md5,'fdt_full_bytes':size,'sampling_rate':fs,'channels':channels,'ecg_index':ecg[0],'eeg_indices':eeg_indices,'source_timestamp_max_error_seconds':error,'header_file':str(hpath)})
        for index,event in enumerate(events):
            if str(event.get('type'))!='stm':continue
            label=str(event.get('label',''));matches=[r for _,r in ratings.iterrows() if label==Path(str(r.stimuliName)).stem or re.fullmatch(re.escape(Path(str(r.stimuliName)).stem)+r'_\d+',label)]
            if len(matches)!=1:excluded.append({'subject':subject,'event':index,'reason':'Ambiguous stimulus/rating mapping'});continue
            row=matches[0]
            try:valence=float(row.valence)
            except (ValueError,TypeError):continue
            if not np.isfinite(valence) or not 1<=valence<=9:continue
            if not (valence<=4 or valence>=6):continue
            following=[];crossed_boundary=False
            for later in events[index+1:]:
                if str(later.get('type'))=='stm':break
                if str(later.get('type'))=='boundary':crossed_boundary=True
                if str(later.get('type'))=='vlnc':following.append(later);break
            if not following:excluded.append({'subject':subject,'event':index,'reason':'No explicit stimulus-ending rating event'});continue
            if crossed_boundary:excluded.append({'subject':subject,'event':index,'reason':'Recording boundary inside stimulus'});continue
            start=float(event['latency'])-1;end=float(following[0]['latency'])-1
            try:trial_clock=abs((end-start)/fs-(datetime.fromisoformat(str(following[0]['begintime']))-datetime.fromisoformat(str(event['begintime']))).total_seconds())
            except (ValueError,TypeError,KeyError):excluded.append({'subject':subject,'event':index,'reason':'Missing source timing verification'});continue
            if trial_clock>=.03:excluded.append({'subject':subject,'event':index,'reason':'Trial timestamps/sample durations disagree'});continue
            count=int(round(60*fs))
            if end-start<count or start<0 or end>points:excluded.append({'subject':subject,'event':index,'reason':'No full 60s stimulus interval'});continue
            first=int(np.ceil((start+end-count)/2));last=first+count
            assert first>=start and last<=end
            offset=first*channels*4;length=count*channels*4;key=f'{subject}_event{index:03d}'
            trials.append({'id':key,'subject':subject,'stimulus':Path(str(row.stimuliName)).stem,'valence':valence,'label':'low' if valence<=4 else 'high','fdt_path':fdt,'fdt_full_bytes':size,'byte_first':offset,'byte_last':offset+length-1,'window_bytes':length,'first_sample':first,'sample_count':count,'sampling_rate':fs,'channels':channels,'ecg_index':ecg[0],'eeg_indices':eeg_indices,'trial_clock_error_seconds':trial_clock,'recording_boundary_inside':False,'file':str(directory/(key+'.f32'))})
        print(json.dumps({'subject':subject,'planned_windows_so_far':len(trials),'full_recording_MB':size/1e6,'verified_event_timing_seconds':error}),flush=True)
    total=sum(t['window_bytes'] for t in trials)
    assert len(subjects)==12 and total<4_000_000_000
    plan={'protocol':PROTOCOL,'subjects':subjects,'trials':trials,'recording_audit':audit,'excluded':excluded,'planned_signal_bytes':total,'free_bytes':shutil.disk_usage(ROOT).free}
    write(ROOT/'reports/dens_download_plan.json',plan)
    if len(audit)<6:
        print('Insufficient dimension-consistent participants for the fixed protocol; no signal ranges requested.',flush=True)
        return
    def fetch(t):
        dest=Path(t['file'])
        if dest.exists() and dest.stat().st_size==t['window_bytes']:return {'id':t['id'],'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
        for attempt in range(4):
            try:
                r=requests.get(S3+t['fdt_path'],headers={'Range':f'bytes={t["byte_first"]}-{t["byte_last"]}'},timeout=(20,90));r.raise_for_status()
                assert r.status_code==206 and r.headers.get('Content-Range')==f'bytes {t["byte_first"]}-{t["byte_last"]}/{t["fdt_full_bytes"]}' and len(r.content)==t['window_bytes']
                temporary=dest.with_suffix('.partial');temporary.write_bytes(r.content);temporary.replace(dest)
                return {'id':t['id'],'bytes':len(r.content),'sha256':hashlib.sha256(r.content).hexdigest()}
            except Exception:
                if attempt==3:raise
                time.sleep(2**attempt)
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for done in as_completed([pool.submit(fetch,t) for t in trials]):results.append(done.result());print('Verified range '+results[-1]['id'],flush=True)
    write(ROOT/'reports/dens_download_manifest.json',{'protocol':PROTOCOL,'windows':results,'total_bytes':sum(r['bytes'] for r in results),'full_fdt_md5_verified':False})
if __name__=='__main__':main()
