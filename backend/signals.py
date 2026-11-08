"""Offline extensions. Features are not validated emotion or disease estimates."""
from typing import Protocol
import numpy as np
from scipy.signal import welch

class SensorAdapter(Protocol):
    """Future adapters provide samples, sample rate, units and sensor provenance."""
    def read(self) -> dict: ...

def finite_array(values,min_count=4):
    x=np.asarray(values,dtype=float)
    if x.size<min_count or not np.isfinite(x).all(): raise ValueError('Signal must contain enough finite numeric samples')
    return x

def eeg_features(samples,sample_rate=128.,sample=False):
    import mne
    x=finite_array(samples)
    if x.ndim==1: x=x[None,:]
    if x.ndim!=2 or x.shape[0]>64: raise ValueError('EEG format must be channels x samples, at most 64 channels')
    if sample_rate<80 or sample_rate>4096: raise ValueError('EEG sampling rate must be 80..4096 Hz for theta/alpha/beta')
    if x.shape[-1]<sample_rate*4: raise ValueError('At least four seconds of EEG required')
    if np.any(np.std(x,axis=-1)<1e-12): raise ValueError('Flat EEG channel')
    raw=mne.io.RawArray(x,mne.create_info([f'EEG{i+1}' for i in range(len(x))],sample_rate,ch_types='eeg'),verbose=False)
    filtered=raw.copy().filter(1.,35.,method='iir',verbose=False).get_data()
    freq,power=welch(filtered,fs=sample_rate,nperseg=min(int(sample_rate*4),x.shape[-1]),axis=-1)
    total=np.trapz(power[:,(freq>=1)&(freq<=35)],freq[(freq>=1)&(freq<=35)],axis=-1)
    bands={}
    for name,(lo,hi) in {'theta':(4,8),'alpha':(8,13),'beta':(13,30)}.items():
        sel=(freq>=lo)&(freq<hi)
        bands[name]={'absolute_v2':np.trapz(power[:,sel],freq[sel],axis=-1).tolist(),'relative':(np.trapz(power[:,sel],freq[sel],axis=-1)/np.maximum(total,1e-30)).tolist()}
    return {'source':'synthetic sample' if sample else 'uploaded offline EEG','sample_data':sample,'sample_rate':sample_rate,'channels':len(x),'duration_seconds':x.shape[-1]/sample_rate,'bands':bands,'basic_quality_pass':bool(np.max(np.ptp(x,axis=-1))<=500e-6),'peak_to_peak_uv':(np.ptp(x,axis=-1)*1e6).tolist(),'quality':'Only a basic 500-microvolt amplitude screen. No ocular/muscle artifact rejection or calibrated emotion model. Input units must be volts.','arousal_hint':None}

def hrv_features(rr_ms=None,signal=None,sample_rate=250.,kind='ecg',sample=False):
    method='RR intervals'
    if rr_ms is None:
        import neurokit2 as nk
        x=finite_array(signal,100)
        if x.ndim!=1: raise ValueError('Physiological waveform must be a single channel')
        if sample_rate<50 or sample_rate>4096: raise ValueError('Waveform sample rate must be 50..4096 Hz')
        if len(x)/sample_rate<30: raise ValueError('At least 30 seconds of ECG/PPG required')
        if kind=='ecg':
            _,info=nk.ecg_process(x,sampling_rate=sample_rate)
            peaks=info['ECG_R_Peaks']
        elif kind=='ppg':
            _,info=nk.ppg_process(x,sampling_rate=sample_rate)
            peaks=info['PPG_Peaks']
        else: raise ValueError('kind must be ecg or ppg')
        rr_ms=np.diff(peaks)/sample_rate*1000
        method=f'NeuroKit2 {kind} peak detection'
    rr=finite_array(rr_ms,20)
    if rr.ndim!=1 or np.any((rr<300)|(rr>2000)): raise ValueError('RR must be a one-dimensional series in milliseconds between 300 and 2000; correct artifacts first')
    return {'source':'synthetic sample' if sample else 'uploaded offline physiology','sample_data':sample,'method':method,'beats':len(rr)+1,'duration_seconds':float(rr.sum()/1000),'rmssd_ms':float(np.sqrt(np.mean(np.diff(rr)**2))),'sdnn_ms':float(np.std(rr,ddof=1)),'mean_hr_bpm':float(60000/rr.mean()),'quality':'Short-window time-domain estimates. No autonomic diagnosis or emotion inference. No ectopic-beat correction.','arousal_hint':None}

def sample_signals():
    rng=np.random.default_rng(42)
    t=np.arange(128*16)/128
    eeg=(15e-6*np.sin(2*np.pi*10*t)+5e-6*np.sin(2*np.pi*6*t)+3e-6*rng.normal(size=len(t))).tolist()
    rr=(850+30*np.sin(np.arange(90)*.3)+rng.normal(0,8,90)).tolist()
    return {'eeg':{'samples':[eeg],'sample_rate':128,'sample_data':True},'hrv':{'rr_ms':rr,'sample_data':True},'disclosure':'Deterministic simulated signals, no sensors connected'}

def fuse_signals(eeg=None,hrv=None):
    """Bounded optional heuristic. No valence is inferred from physiology."""
    hints=[]
    if eeg:
        alpha=float(np.mean(eeg['bands']['alpha']['relative']))
        beta=float(np.mean(eeg['bands']['beta']['relative']))
        hints.append(beta>alpha*1.5)
    if hrv: hints.append(hrv['mean_hr_bpm']>100)
    return {'arousal_hint':'high' if hints and any(hints) else None,'available_modalities':[k for k,v in [('eeg',eeg),('hrv',hrv)] if v],'interpretation':'Exploratory fixed rule, not validated affect inference'}
