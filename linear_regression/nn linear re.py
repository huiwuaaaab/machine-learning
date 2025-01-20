import torch.nn as nn
import torch.utils.data as data
import torch

def Datacreate(w,b,num_sample):
    x=torch.randn(num_sample,len(w))
    y=torch.randn(num_sample,1)
    return x,y

def Tensordataloader(dataall,batch_size):
    dataset=data.TensorDataset(*dataall)
    return data.DataLoader(dataset,batch_size,shuffle=True)


W=torch.tensor([2.5,1.6],requires_grad=True)
bias=torch.zeros(len(W))
feature,label=Datacreate(W,bias,1000)
dataloader=Tensordataloader((feature,label),50)
net=nn.Linear(2,1)
loss=nn.MSELoss()
net.weight.data.normal_(0,0.01)
net.bias.data.fill_(0)
trainer=torch.optim.SGD((net.weight,net.bias),lr=0.001)
for epoch in range(3):
    for x,y in dataloader:
        l=loss(net(x),y)
        l.backward()
        trainer.step()
        trainer.zero_grad()
    l=loss(net(feature),label)
    print('%s loss:%f'%(epoch,l))