from softmax_cls.softmax_classification import ImageData
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


class Reshape(nn.Module):
    def forward(self,x):
        return x.view(-1,1,28,28)

def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.sum())

class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_accuracy(net, data_iter):  #@save
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            X,y=X.cuda(),y.cuda()
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def plt_acc(train,val,epoch):
    plt.plot(range(epoch),train)
    plt.plot(range(epoch),val)
    plt.legend(['train','val'])
    plt.title('train_val accuracy')
    plt.show()

def gpu_train(net,train_iter,test_iter,epochs,lr,momentum=0,weightdecay=0):
    net.to(device='cuda')
    optimizer=torch.optim.SGD(net.parameters(),lr=lr,momentum=momentum,weight_decay=weightdecay)
    loss=nn.CrossEntropyLoss(reduction='mean')
    train_ls=[]
    val_ls=[]
    for epoch in tqdm(range(epochs)):
        net.train()
        for x,y in train_iter:
            x,y=x.cuda(),y.cuda()
            y_pred=net(x)
            l=loss(y_pred,y)
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
        with torch.no_grad():
            train_acc=evaluate_accuracy(net,train_iter)
            eval_acc=evaluate_accuracy(net,test_iter)
            train_ls.append(train_acc)
            val_ls.append(eval_acc)
    plt_acc(train_ls,val_ls,epochs)

def init_net(layer):
    if layer==nn.Linear or layer==nn.Conv2d:
        nn.init.xavier_uniform(layer)

if __name__=='__main__':
    net = nn.Sequential(Reshape(), nn.Conv2d(1, 6, 5, padding=2), nn.Sigmoid(),
                        nn.AvgPool2d(2),
                        nn.Conv2d(6, 16, 5), nn.Sigmoid(),
                        nn.AvgPool2d(2), nn.Flatten(),
                        nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
                        nn.Linear(120, 84), nn.Sigmoid(),
                        nn.Linear(84, 10))
    net.apply(init_net)

    image=ImageData()
    train,test=image.loadImage()
    trainloader=image.ImgdataLoader(train,50)
    testloader=image.ImgdataLoader(test,50)
    gpu_train(net,trainloader,testloader,25,0.9)