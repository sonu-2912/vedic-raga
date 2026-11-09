"""Paper-informed mechanisms, with explicit implementation choices. CPU only."""
import torch
from torch import nn
from torch.nn import functional as F

FER_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


class DenseBlock(nn.Module):
    def __init__(self, channels, growth=12, layers=3, dilated=False):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(layers):
            d = 2 if dilated and i % 2 else 1
            self.layers.append(nn.Sequential(nn.BatchNorm2d(channels+i*growth), nn.ReLU(),
                nn.Conv2d(channels+i*growth, growth, 3, padding=d, dilation=d, bias=False)))
        self.out_channels = channels + growth*layers

    def forward(self, x):
        for layer in self.layers:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class FusionDenseBlock(nn.Module):
    """Eqs 3-8: two image views, gated cross attention, L2 weights, concatenation.

    At is instantiated as scaled dot-product multihead attention over 8x8 grids.
    This operator, channel sizes and two visible views are assumptions.
    """
    def __init__(self, channels=24, attention=True, dilated=True, version=1):
        super().__init__()
        self.a = DenseBlock(channels, dilated=dilated)
        self.b = DenseBlock(channels, dilated=dilated)
        self.proj_a = nn.Conv2d(self.a.out_channels, 32, 1)
        self.proj_b = nn.Conv2d(self.b.out_channels, 32, 1)
        self.cross = nn.MultiheadAttention(32, 4, batch_first=True)
        self.use_attention = attention
        self.reduce = nn.Conv2d(128, 48, 1)
        self.activation = nn.Sequential(nn.BatchNorm2d(48),nn.ReLU()) if version>=2 else nn.Identity()

    def forward(self, a, b):
        a = F.adaptive_avg_pool2d(self.proj_a(self.a(a)), (8,8))
        b = F.adaptive_avg_pool2d(self.proj_b(self.b(b)), (8,8))
        if self.use_attention:
            qa, qb = a.flatten(2).transpose(1,2), b.flatten(2).transpose(1,2)
            kv = torch.cat([qa,qb],1)
            ba = self.cross(qa, kv, kv, need_weights=False)[0].transpose(1,2).reshape_as(a)
            bb = self.cross(qb, kv, kv, need_weights=False)[0].transpose(1,2).reshape_as(b)
            # Eq 7 L2 normalization with epsilon. Sigmoid interprets maps as gates.
            ga = torch.sigmoid(F.normalize(ba,dim=1,eps=1e-6))
            gb = torch.sigmoid(F.normalize(bb,dim=1,eps=1e-6))
        else:
            ga, gb = torch.ones_like(a), torch.ones_like(b)
        pa=F.avg_pool2d(ga*a,3,stride=1,padding=1)
        pb=F.avg_pool2d(gb*b,3,stride=1,padding=1)
        return self.activation(self.reduce(torch.cat([a,b,pa,pb],1)))


class FusDenseNet(nn.Module):
    def __init__(self, classes=7, attention=True, dilated=True, version=1):
        super().__init__()
        self.version=version
        self.stem = nn.Sequential(nn.Conv2d(1,24,3,padding=1), nn.ReLU(), nn.AvgPool2d(2))
        self.fusion = FusionDenseBlock(attention=attention,dilated=dilated,version=version)
        # Fig 2 repeats fusion blocks with cross-dense concatenation. Three stages
        # are a concrete choice for its schematic ellipsis, not a recovered depth.
        self.more_fusion = nn.ModuleList([FusionDenseBlock(attention=attention,dilated=dilated,version=version) for _ in range(2)])
        self.cross_connections = nn.ModuleList([nn.Conv2d(72,24,1),nn.Conv2d(120,24,1)])
        self.final_fusion = nn.Conv2d(168,48,1)
        self.final_activation=nn.Sequential(nn.BatchNorm2d(48),nn.ReLU()) if version>=2 else nn.Identity()
        self.dense = DenseBlock(48)
        c = self.dense.out_channels
        # Project variant 3 retains the paper mechanisms and adds a spatial
        # shortcut from the 24x24 stem before attention's 8x8 pooling bottleneck.
        # Early mean/max 4x4 pooling contributes 768 texture/location features.
        self.classifier = nn.Sequential(nn.Linear(c*(7 if version>=2 else 3)+(768 if version>=3 else 0),128 if version>=2 else 64),nn.ReLU(),nn.Dropout(.35 if version>=2 else .2),nn.Linear(128 if version>=2 else 64,classes))
        self.reconstruct = nn.Conv2d(48,1,1)

    def forward(self,x,second_view=None,return_reconstruction=False):
        reconstruction_target_size=x.shape[-2:]
        if self.version>=2:
            x=x*2-1
            if second_view is not None:second_view=second_view*2-1
        if second_view is None:
            second_view = F.avg_pool2d(x,3,stride=1,padding=1)
        a,b=self.stem(x),self.stem(second_view)
        if self.version>=3:
            early=(a+b)/2
            spatial=torch.cat([F.adaptive_avg_pool2d(early,(4,4)).flatten(1),F.adaptive_max_pool2d(early,(4,4)).flatten(1)],1)
            spatial=F.layer_norm(spatial,(768,))
        history=[F.adaptive_avg_pool2d((a+b)/2,(8,8)),self.fusion(a,b)]
        for project,block in zip(self.cross_connections,self.more_fusion):
            view=project(torch.cat(history,1))
            history.append(block(view,F.avg_pool2d(view,3,stride=1,padding=1)))
        fused=self.final_activation(self.final_fusion(torch.cat(history,1)))
        f = self.dense(fused)
        pooled = torch.cat([f.mean((2,3)),f.amax((2,3)),torch.sqrt(f.square().mean((2,3))+1e-6)],1)
        if self.version>=2:pooled=torch.cat([pooled,F.adaptive_avg_pool2d(f,(2,2)).flatten(1)],1)
        if self.version>=3:pooled=torch.cat([pooled,spatial],1)
        logits = self.classifier(pooled)
        if return_reconstruction:
            recon = F.interpolate(self.reconstruct(fused),size=reconstruction_target_size,mode='bilinear',align_corners=False)
            if self.version>=2:recon=torch.sigmoid(recon)
            return logits,recon
        return logits


class FDCF(nn.Module):
    """Eqs 10-15. Features and preference windows are project choices."""
    def __init__(self, users, ragas, emotions=7, dim=16):
        super().__init__()
        self.user = nn.Embedding(users,dim)
        self.music = nn.Embedding(ragas,dim)
        self.emotion = nn.Embedding(emotions,dim)
        self.representation = nn.Linear(dim*3,dim)
        self.focus = nn.Linear(dim*3,2)
        self.mlp = nn.Sequential(nn.Linear(dim*2,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))

    def forward(self,users,ragas,emotions,short=None,long=None,return_details=False):
        u,m,e = self.user(users),self.music(ragas),self.emotion(emotions)
        short = u if short is None else short
        long = u if long is None else long
        encoded = self.representation(torch.cat([u,m,e],-1))
        weights = torch.softmax(self.focus(torch.cat([encoded,long,short],-1)),dim=-1)
        beta = weights[...,:1]
        dynamic = beta*short+(1-beta)*long
        interaction = e*m
        logits = self.mlp(torch.cat([dynamic,interaction],-1)).squeeze(-1)
        return (logits,{'beta':beta,'dynamic':dynamic,'interaction':interaction}) if return_details else logits
