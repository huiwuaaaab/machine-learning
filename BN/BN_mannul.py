import torch
import torch.nn as nn

def norm(x,u,o,a,b,eps):
    if not torch.enable_grad():
        x1=(x-u)/torch.sqrt(o+eps)
    else:
        assert len(x.shape) in [2,4]
        if len(x.shape)==2:
            u=torch.mean(x,dim=0,keepdim=True)
            o=torch.var(x,dim=0,keepdim=True)
        elif len(x.shape)==4:
            u=torch.mean(x,dim=(0,2,3),keepdim=True)
            o=torch.var(x,dim=(0,2,3),keepdim=True)
        x1=(x-u)/torch.sqrt(o+eps)
    u+=u
    u=u.mean(dim=0,keepdim=True)
    o+=o
    o=o.mean(dim=(0,2,3),keepdim=True)
    x1=a*x1+b
    return x1,u,o

class BN(nn.Module):
    def __init__(self,feature,dim):
        super().__init__()
        if dim == 2:
            shape = (1, feature)
        if dim == 4:
            shape = (1, feature, 1, 1)
        self.a=nn.Parameter(torch.zeros(shape))
        self.b=nn.Parameter(torch.ones(shape))

    def forward(self,x,motionu,motiono):
        if motionu.device!=x.device:
            motionu=motionu.to(x.device)
            motiono=motiono.to(x.device)
        y,motionu,motiono=norm(x,motionu,motiono,self.a,self.b,1e-5)
        return y,motionu,motiono