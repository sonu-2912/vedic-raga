"""Local inference. Does not send or persist images. Checkpoints use weights_only."""
import io,threading,json,hashlib
import cv2
import numpy as np
import torch
from torch import nn
from PIL import Image,ImageOps,UnidentifiedImageError
from torchvision import models,transforms
from .models import FusDenseNet,FER_CLASSES
from .inference import emotion_logits
from .face_policy import expression_suggestion
from .catalog import ROOT,RASA_EMOTIONS

torch.set_num_threads(2)
NAMES=['adhbuta','bhayanak','bhibhatsa','hasya','karuna','raudra','shanta','shringara','veer']
_models={}
_infos={}
_versions={}
_lock=threading.Lock()

def denoise(image,mode):
    if mode=='none': return image
    if mode in ('nlm','hybrid'): image=cv2.fastNlMeansDenoising(image,None,5,7,21)
    if mode in ('bilateral','hybrid'): image=cv2.bilateralFilter(image,5,30,30)
    if mode in ('gaussian','hybrid'): image=cv2.GaussianBlur(image,(3,3),.6)
    return image

def load_model(name):
    with _lock:
        path=ROOT/('models/fusdense.pt' if name=='fusdense' else 'external/raag-ratnakar-main/best_navarasa_model.pth')
        version=(path.stat().st_mtime_ns,path.stat().st_size) if path.exists() else None
        if name in _models and _versions.get(name)==version:return _models[name]
        if name=='fusdense':
            path=ROOT/'models/fusdense.pt'
            if not path.exists(): raise RuntimeError('FusDenseNet has no trained weights. Train with FER2013 or use the supplied Navrasa baseline.')
            checkpoint_bytes=path.read_bytes()
            ck=torch.load(io.BytesIO(checkpoint_bytes),map_location='cpu',weights_only=True)
            if ck.get('synthetic',False): raise RuntimeError('Synthetic smoke weights cannot be used as a facial emotion model.')
            model=FusDenseNet(**ck.get('architecture',{}));model.load_state_dict(ck['state_dict'])
            names=ck['classes']
            if names != FER_CLASSES: raise RuntimeError('FER checkpoint class order mismatch')
            _infos[name]={'pilot':ck.get('pilot',False),'epoch':ck.get('epoch'),'train_count':ck.get('train_count'),'architecture':ck.get('architecture',{}),'inference':ck.get('inference',{}),'checkpoint_sha256':hashlib.sha256(checkpoint_bytes).hexdigest(),'pure_fusdense':not any(k.startswith('transfer_') for k in ck['state_dict']),'external_encoder':any(k.startswith('transfer_') for k in ck['state_dict']),'training_stage':ck.get('training_stage')}
        else:
            if not path.exists():raise RuntimeError('Optional Navrasa baseline weights are unavailable. Provide the original best_navarasa_model.pth under external/raag-ratnakar-main; the trained FusDenseNet remains available.')
            model=models.efficientnet_b2(weights=None)
            model.classifier=nn.Sequential(nn.Dropout(.4),nn.Linear(1408,512),nn.ReLU(),nn.Dropout(.3),nn.Linear(512,9))
            ck=torch.load(ROOT/'external/raag-ratnakar-main/best_navarasa_model.pth',map_location='cpu',weights_only=True)
            state=ck.get('model_state_dict',ck)
            model.load_state_dict(state)
            names=ck.get('class_names',NAMES)
            aliases={'adhbut':'adhbuta','bhayank':'bhayanak','sringara':'shringara','bibhatsa':'bhibhatsa','vira':'veer','adbhuta':'adhbuta'}
            names=[aliases.get(n.lower().strip(),n.lower().strip()) for n in names]
            if set(names)!=set(NAMES): raise RuntimeError(f'Unrecognized checkpoint classes: {names}')
        model.eval()
        if name=='fusdense':model._ragamood_info=dict(_infos[name])
        _models[name]=(model,names)
        _versions[name]=version
        return model,names

def predict(raw,model_name='fusdense',denoising='none',require_face=True):
    if len(raw)>8*1024*1024: raise ValueError('Image exceeds 8 MB')
    try:
        im=Image.open(io.BytesIO(raw))
        if im.width*im.height>20_000_000: raise ValueError('Image exceeds 20 megapixels')
        im=ImageOps.exif_transpose(im).convert('RGB')
    except (UnidentifiedImageError,OSError) as e: raise ValueError('Invalid image') from e
    arr=np.array(im)
    gray=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)
    detector=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    faces=detector.detectMultiScale(gray,1.1,5,minSize=(36,36))
    bbox=None
    if len(faces):
        x,y,w,h=map(int,max(faces,key=lambda f:f[2]*f[3]));bbox=[x,y,w,h]
        arr=arr[y:y+h,x:x+w];gray=gray[y:y+h,x:x+w]
    elif require_face: raise ValueError('No frontal face detected. Improve lighting, face the camera, or use manual mood input.')
    model,names=load_model(model_name)
    info=getattr(model,'_ragamood_info',{})
    if model_name=='fusdense':
        x=transforms.ToTensor()(Image.fromarray(cv2.resize(denoise(gray,denoising),(48,48))))
    else:
        if denoising!='none': arr=cv2.cvtColor(denoise(gray,denoising),cv2.COLOR_GRAY2RGB)
        x=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])(Image.fromarray(arr))
    with torch.inference_mode():probs=emotion_logits(model,x.unsqueeze(0),info.get('inference',{})).softmax(1)[0].tolist() if model_name=='fusdense' else model(x.unsqueeze(0)).softmax(1)[0].tolist()
    original=dict(zip(names,probs))
    emotion_probs={n:0. for n in FER_CLASSES}
    if model_name=='fusdense': emotion_probs=original
    else:
        for name,p in original.items(): emotion_probs[RASA_EMOTIONS[name]]+=p
    emotion=max(emotion_probs,key=emotion_probs.get)
    evaluation=None
    if model_name=='fusdense':
        report_path=ROOT/'reports/emotion_evaluation.json'
        if report_path.exists():
            report=json.loads(report_path.read_text())
            if report.get('checkpoint_sha256')==info['checkpoint_sha256']:
                evaluation={k:report[k] for k in ['n','accuracy','pilot','checkpoint_epoch']}
                evaluation['macro_f1']=report['classification_report']['macro avg']['f1-score']
                evaluation['checkpoint_sha256']=report['checkpoint_sha256']
        validation='Limited real-data FER2013 pilot; no exact reproduction or clinical validity claimed.' if info['pilot'] else 'Project-trained model. Check matching evaluation provenance before interpreting performance.'
    else:validation='No held-out native Navrasa dataset supplied; mapped FER performance unverified'
    return {'pure_fusdense':bool(info.get('pure_fusdense',False)),'external_encoder':bool(info.get('external_encoder',False)),'model_variant':('FusDenseNet + ImageNet DenseNet121 probability ensemble' if info.get('architecture',{}).get('version',0)==7 else 'FusDenseNet + frozen ImageNet DenseNet121 hybrid' if info.get('architecture',{}).get('version',0)==6 else 'Pure FusDenseNet' if model_name=='fusdense' else 'Navrasa baseline'),'emotion':emotion,'confidence':emotion_probs[emotion],'probabilities':emotion_probs,'native_probabilities':original,'native_label':names[int(np.argmax(probs))],'model':model_name,'trained':True,'face_box':bbox,'low_confidence':emotion_probs[emotion]<.5,'label_mapping':'FER2013 native labels' if model_name=='fusdense' else 'Project heuristic mapping of nine Navrasa labels to seven FER labels; not FER calibration','validation':validation,'held_out_evaluation':evaluation,**expression_suggestion(emotion_probs,info.get('checkpoint_sha256'))}
