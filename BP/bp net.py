import torch

class Activate(object):
    def __init__(self):
        pass

    def sigmod(self,x):
        return 1/(1+torch.exp(x))

    def relu(self,X):
        a=torch.zeros_like(X)
        return torch.max(X,a)


def Datacreate(num_sample,input,output):
        feature=torch.randn((num_sample,input))
        label=torch.randn(num_sample,output)
        return feature,label


def wb_init(input,layer,output):
    w1=torch.randn((input,layer),requires_grad=True)
    b1=torch.randn((1,layer),requires_grad=True)
    w2=torch.randn((layer,output),requires_grad=True)
    b2=torch.randn((1,output),requires_grad=True)
    return w1,b1,w2,b2

def forwanrd(w,b,x):
    return torch.matmul(x,w)+b

def loss(y_pred,y):
    return (y_pred-y)**2

if __name__=='__main__':
    num=1000
    x,y=Datacreate(num,256,20)
    activator=Activate()
    w1,b1,w2,b2=wb_init(256,128,20)
    for epoch in range(10):
        l=0
        for row in x:
            x1=forwanrd(w1,b1,row)
            x1_a=activator.relu(x1)
            x2=forwanrd(w2,b2,x1_a)
            l+=loss(x2,y)
        l.mean().backward()
        with torch.no_grad():
            for param in [w1,b1,w2,b2]:
                param-=0.000001*param.grad
                param.grad.zero_()
        print('loss:%f'%l.mean())