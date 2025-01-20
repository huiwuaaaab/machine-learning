import random
import torch
def Datacreate(w,b,num_sample):
    x=torch.randn(num_sample,len(w))
    y=torch.randn(num_sample,1)
    return x,y

def Tensordataloader(x,y,batch_size):
    index=[u for u in range(len(x))]
    random.shuffle(index)
    for i in (0,len(x),batch_size):
        index_b=torch.tensor(index[i:min(i+batch_size,len(x))],dtype=int)
        yield x[index_b],y[index_b]

def Net(x,w,b):
    return torch.matmul(x,w)+b

W=torch.tensor([2.3,4.6],requires_grad=True)
bias=torch.zeros_like(W,requires_grad=True)
feature,label=Datacreate(W,bias,1000)
lr=0.1
for epoch in range(5):
    for batch in range(5):
        x,y=next(Tensordataloader(feature,label,12))
        y_pred=Net(x,W,bias)
        loss=(y_pred-y)**2
        loss=loss.mean()
        loss.backward()
        with torch.no_grad():
            W-= lr * W.grad
            bias -= lr * bias.grad
            W.grad.zero_()
            bias.grad.zero_()
    with torch.no_grad():
        loss_all=(Net(feature,W,bias)-label)**2
        print('{0}:loss {1}'.format(epoch,loss_all.mean()))