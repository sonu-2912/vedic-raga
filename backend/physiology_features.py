"""Real-recording features for the GamER experiment, not a mood diagnosis."""
import numpy as np
from scipy import signal

EEG_NAMES=['theta_relative','alpha_relative','beta_relative','delta_relative','gamma_relative','spectral_entropy','hjorth_mobility','hjorth_complexity']
CARDIAC_NAMES=['mean_rr_ms','rmssd_ms','sdnn_ms','pnn50','median_rr_ms','rr_iqr_ms']

def paired_features(eeg, ecg, sample_rate):
    """Scale-invariant EEG features and beat intervals from the same time slice.

    Windows may be ultra-short: the derived RR statistics are research features,
    not clinically validated HRV estimates. No learned preprocessing here.
    """
    import neurokit2 as nk
    eeg=np.asarray(eeg,dtype=float).flatten();ecg=np.asarray(ecg,dtype=float).flatten()
    fs=float(sample_rate)
    if len(eeg)!=len(ecg) or len(eeg)<15*fs:raise ValueError('At least 15 seconds of paired data required')
    if not np.isfinite(eeg).all() or not np.isfinite(ecg).all():raise ValueError('Non-finite recording')
    if np.std(eeg)<1e-8 or np.std(ecg)<1e-8:raise ValueError('Flat recording')
    # Hardware ADC saturation test. Physically scaled inputs are screened upstream.
    for raw in [eeg,ecg]:
        if np.mean((raw<=1)|(raw>=65534))>.02 and np.max(raw)>100:raise ValueError('ADC saturation')
    filtered=signal.sosfiltfilt(signal.butter(4,[1,40],btype='bandpass',fs=fs,output='sos'),eeg)
    frequencies,power=signal.welch(filtered,fs=fs,nperseg=min(len(eeg),int(4*fs)))
    def area(low,high):
        keep=(frequencies>=low)&(frequencies<high)
        return float(np.trapz(power[keep],frequencies[keep]))
    total=max(area(1,40),1e-18)
    spectral=power[(frequencies>=1)&(frequencies<40)];spectral=spectral/max(spectral.sum(),1e-18)
    mobility=np.sqrt(np.var(np.diff(filtered))/max(np.var(filtered),1e-18))
    derivative_mobility=np.sqrt(np.var(np.diff(filtered,2))/max(np.var(np.diff(filtered)),1e-18))
    eeg_values=[area(a,b)/total for a,b in [(4,8),(8,13),(13,30),(1,4),(30,40)]]
    eeg_values += [float(-np.sum(spectral*np.log(spectral+1e-18))/np.log(len(spectral))),float(mobility),float(derivative_mobility/max(mobility,1e-18))]
    cleaned=nk.ecg_clean(ecg,sampling_rate=fs,method='neurokit')
    _,info=nk.ecg_peaks(cleaned,sampling_rate=fs,method='neurokit',correct_artifacts=False)
    rr=np.diff(info['ECG_R_Peaks'])/fs*1000
    valid=(rr>=300)&(rr<=2000)
    if len(rr)<8 or valid.mean()<.9:raise ValueError('Too few credible RR intervals')
    # Do not silently concatenate gaps for successive-difference statistics.
    differences=np.diff(rr)[valid[1:]&valid[:-1]]
    clean_rr=rr[valid]
    if len(differences)<6:raise ValueError('Too few consecutive valid beats')
    cardiac=[float(clean_rr.mean()),float(np.sqrt(np.mean(differences**2))),float(np.std(clean_rr,ddof=1)),float(np.mean(np.abs(differences)>50)),float(np.median(clean_rr)),float(np.subtract(*np.percentile(clean_rr,[75,25])))]
    return eeg_values,cardiac,{'duration_seconds':len(eeg)/fs,'rr_count':int(len(rr)),'valid_rr_fraction':float(valid.mean()),'ultrashort_hrv':len(eeg)/fs<60}
