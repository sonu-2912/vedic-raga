"""Provision the optional, rejected teacher-screen resource with exact SHA checks.

This is not the application's FusDenseNet or a default app dependency.
"""
import hashlib,json,shutil
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]
COMMIT='48b10fa505aeb51fbd127fc60a5c660040e823b7'
SHA='4f61307602fc089ce20488a31d4e4614e3c9753a7d6c41578c854858b183e1a9'
SIZE=4791892

def main():
    folder=ROOT/'external/opencv_fer_teacher';folder.mkdir(parents=True,exist_ok=True)
    model=folder/'mobilefacenet.onnx'
    if model.exists():
        if hashlib.sha256(model.read_bytes()).hexdigest()!=SHA:raise ValueError('Existing model differs; preserve and inspect it, do not overwrite.')
        print('Verified optional teacher already present. It is not the FusDenseNet app model.');return
    if shutil.disk_usage(ROOT).free<100_000_000:raise RuntimeError('At least100MB free space required')
    raw=f'https://raw.githubusercontent.com/opencv/opencv_zoo/{COMMIT}/'
    for remote,local in [('LICENSE','LICENSE.txt'),('models/facial_expression_recognition/README.md','README.md'),('models/facial_expression_recognition/facial_fer_model.py','upstream_inference.py')]:
        dest=folder/local
        if not dest.exists():
            r=requests.get(raw+remote,timeout=30);r.raise_for_status();dest.write_bytes(r.content)
    url=f'https://media.githubusercontent.com/media/opencv/opencv_zoo/{COMMIT}/models/facial_expression_recognition/facial_expression_recognition_mobilefacenet_2022july.onnx'
    with requests.get(url,stream=True,timeout=(20,90)) as r:
        r.raise_for_status();parts=[];total=0
        for chunk in r.iter_content(65536):
            total+=len(chunk)
            if total>SIZE:raise ValueError('Unexpected model size')
            parts.append(chunk)
    data=b''.join(parts)
    if len(data)!=SIZE or hashlib.sha256(data).hexdigest()!=SHA:raise ValueError('Model checksum mismatch; no model installed')
    model.write_bytes(data)
    print(json.dumps({'model_sha256':SHA,'bytes':SIZE,'licence':'Apache2.0','source_commit':COMMIT,'purpose':'Optional validation-only screen, not FusDenseNet'}))
if __name__=='__main__':main()
