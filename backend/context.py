"""Transparent experimental recommendation context, not an affect classifier."""
def weather_context(weather=None):
    if not weather:return {'available':False,'season_tags':[],'source':None}
    condition=weather.get('condition');temperature=weather.get('temperature_c')
    tags=['varsha'] if condition=='rain' else ['hemanta','shishira'] if condition=='snow' or (temperature is not None and temperature<=10) else ['grishma'] if condition=='hot' or (temperature is not None and temperature>=30) else []
    return {'available':True,'weather':weather,'source':weather.get('source','manual'),'season_tags':tags,'policy':'Weather-inspired preference for matching source Ritu annotations, capped at 0.03. Weather is not geographical season or a mood measurement.','validated':False}

def body_context(eeg=None,hrv=None,baseline=None,allow_sample=False):
    signals={k:v for k,v in [('eeg',eeg),('hrv',hrv)] if v}
    result={'available':bool(signals),'features':signals,'available_modalities':list(signals),'sample_data':any(v.get('sample_data',False) for v in signals.values()),'deviations':[],'calming_context':False,'arousal_hint':None,'validated':False,'interpretation':'Features describe signals; they do not identify happy, disgusted, neutral, stressed or a disease.'}
    if not signals:return result
    if any(v.get('sample_data',False) for v in signals.values()) and any(not v.get('sample_data',False) for v in signals.values()):result['reason']='Simulated and real inputs cannot be combined as one body recording.';return result
    if not baseline:result['reason']='Features attached. A personal reference is needed before relative physiology can affect ranking.';return result
    if (result['sample_data'] or baseline.get('sample_data')) and not allow_sample:result['reason']='Sample values are excluded from personal recommendations unless sample demonstration is explicitly enabled.';return result
    if baseline.get('sample_data') and any(not v.get('sample_data',False) for v in signals.values()):result['reason']='A simulated reference cannot calibrate uploaded real signals.';return result
    if eeg and hrv and not result['sample_data'] and (not eeg.get('session_id') or eeg.get('session_id')!=hrv.get('session_id')):result['reason']='EEG and HRV need the same declared session_id before a joint reference comparison. Unrelated recordings must not be treated as synchronized.';return result
    if eeg and not eeg.get('basic_quality_pass',True):result['reason']='EEG failed a basic amplitude screen; review artifacts before contextual use.';return result
    if eeg and baseline.get('alpha_beta_ratio') and eeg.get('duration_seconds',0)>=10:
        import numpy as np
        alpha=float(np.mean(eeg['bands']['alpha']['relative']));beta=float(np.mean(eeg['bands']['beta']['relative']))
        ratio=alpha/max(beta,1e-9);relative=ratio/baseline['alpha_beta_ratio']
        result['deviations'].append({'feature':'EEG alpha/beta relative to reference','ratio':relative,'below_reference':relative<.8})
    if hrv and baseline.get('rmssd_ms') and hrv.get('duration_seconds',0)>=60:
        relative=hrv['rmssd_ms']/baseline['rmssd_ms']
        result['deviations'].append({'feature':'RMSSD relative to reference','ratio':relative,'below_reference':relative<.8})
    # Require two concordant modalities; do not turn heart rate into an emotion.
    result['calming_context']=len(result['deviations'])>=2 and all(x['below_reference'] for x in result['deviations'])
    result['reason']='Both values are below the supplied references. The experimental policy can suggest a calm listening direction; it does not determine your mood.' if result['calming_context'] else 'No concordant EEG/HRV reference deviation. Available features are retained without an automatic physiology adjustment.'
    return result
