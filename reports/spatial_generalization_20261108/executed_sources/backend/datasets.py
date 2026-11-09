"""FER CSV and Kaggle train/test folder loader. Split names stay explicit."""
from pathlib import Path
import io,zipfile,hashlib
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from .emotion import denoise
from .models import FER_CLASSES

class FER2013(Dataset):
    def __init__(self,path,split='train',augment=False,denoising='none',limit=None,vertical_flip=False):
        self.path=Path(path);self.rows=[];self.denoising=denoising;self.zip=False;self._zip_reader=None
        self._image_cache=None
        self.transform=transforms.Compose(([transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(.2 if vertical_flip else 0),transforms.RandomRotation(15),transforms.RandomResizedCrop(48,scale=(.85,1.))] if augment else [transforms.Resize((48,48))])+[transforms.ToTensor()])
        if self.path.suffix.lower()=='.csv':
            df=pd.read_csv(self.path)
            expected={'emotion','pixels','Usage'}
            if not expected.issubset(df.columns): raise ValueError(f'CSV requires {expected}')
            usage={'train':'Training','val':'PublicTest','test':'PrivateTest'}[split]
            for row in df[df.Usage==usage].itertuples(): self.rows.append((row.pixels,int(row.emotion)))
            self.csv=True
        else:
            # Directory mode: train/, val/ (optional), test/. Split train stratified if no val/.
            folder=self.path/split
            from sklearn.model_selection import train_test_split
            if self.path.suffix.lower()=='.zip':
                self.zip=True
                with zipfile.ZipFile(self.path) as z:names=sorted(z.namelist())
                has_val=any(n.startswith('val/') for n in names)
                source_split='train' if split=='val' and not has_val else split
                rows=[]
                for name in names:
                    parts=Path(name).parts
                    if len(parts)>=3 and parts[0]==source_split and parts[1] in FER_CLASSES and Path(name).suffix.lower() in {'.png','.jpg','.jpeg'}:
                        rows.append((name,FER_CLASSES.index(parts[1])))
            else:
                source=self.path/'train' if split=='val' and not folder.exists() else folder
                if not source.exists(): raise ValueError(f'Missing dataset folder: {source}')
                has_val=(self.path/'val').exists()
                rows=[]
                for label,name in enumerate(FER_CLASSES):
                    for p in sorted((source/name).glob('*')):
                        if p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp'}: rows.append((p,label))
            if split in {'train','val'} and not has_val:
                train,val=train_test_split(rows,test_size=.15,stratify=[y for _,y in rows],random_state=42)
                rows=train if split=='train' else val
            self.rows=rows;self.csv=False
        if limit and limit<len(self.rows):
            from sklearn.model_selection import train_test_split
            if limit<7:raise ValueError('Use --limit of at least 7 to represent all emotion classes')
            indices,_=train_test_split(np.arange(len(self.rows)),train_size=limit,stratify=[y for _,y in self.rows],random_state=42)
            self.rows=[self.rows[int(i)] for i in sorted(indices)]
        if not self.rows: raise ValueError('No images found for requested split')

    def __len__(self): return len(self.rows)
    def raw_image(self,i):
        if self._image_cache is not None:return self._image_cache[i],self.rows[i][1]
        image,label=self.rows[i]
        if not 0<=label<7: raise ValueError('FER label outside 0..6')
        if self.csv:
            values=np.fromstring(image,sep=' ',dtype=np.uint8)
            if len(values)!=2304: raise ValueError('FER pixels must contain 2304 values')
            arr=values.reshape(48,48)
        elif self.zip:
            if self._zip_reader is None:self._zip_reader=zipfile.ZipFile(self.path)
            with Image.open(io.BytesIO(self._zip_reader.read(image))) as im:arr=np.array(im.convert('L'))
        else:
            with Image.open(image) as im: arr=np.array(im.convert('L'))
        return arr,label

    def cache_images(self):
        """Cache only the current split, after deduplication; no test access."""
        self._image_cache=[self.raw_image(i)[0] for i in range(len(self))]

    def fingerprints(self):
        return [hashlib.sha256(self.raw_image(i)[0].tobytes()).hexdigest() for i in range(len(self))]

    def __getitem__(self,i):
        arr,label=self.raw_image(i)
        return self.transform(Image.fromarray(denoise(arr,self.denoising))),label
