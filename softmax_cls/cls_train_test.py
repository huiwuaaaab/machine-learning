from softmax_cls import softmax_classification
import torch.utils.data
import torch.nn as nn


class nn_sofm(object):
    def __init__(self, dataset_minist, data_test):
        self.train = dataset_minist
        self.test = data_test

    def net(self):
        self.Net = nn.Sequential(nn.Flatten(),nn.Linear(784, 10))
        return self.Net

    def loss(self, y, y_pred):
        l=nn.CrossEntropyLoss()
        return l(y_pred,y).mean()


def mannul_sofm(dataset_minist,data_test):
    W = torch.randn((784, 10), requires_grad=True)
    bias = torch.zeros(10, requires_grad=True)
    for epoch in range(10):
        for _,(x,y) in enumerate(dataset_minist):
            linear_re= softmax_classification.Linear_regression()
            y_pred=linear_re.linear(W,bias,x)
            softmax1= softmax_classification.Softmax_Classfy()
            y_sofm=softmax1.softmax(y_pred)
            l=softmax1.loss(y_sofm,y)
            l.backward()
            W,bias=linear_re.upgrade(0.001,W,bias,W.grad,bias.grad)
            W.grad.zero_()
            bias.grad.zero_()
        accuracy=0
        count=1
        for x,y in data_test:
            linear_re = softmax_classification.Linear_regression()
            y_pred = linear_re.linear(W, bias, x)
            softmax1 = softmax_classification.Softmax_Classfy()
            y_sofm = softmax1.softmax(y_pred)
            y_pred_index=y_sofm.argmax(axis=1)
            Accum=y_pred_index.type(y.dtype)==y
            accuracy+=float(Accum.mean(dtype=float))
            count+=1
        accuracy=accuracy/count
        print('Accuracy:%f'%accuracy)

def nn_sofm_train_test(dataset_minist,data_test):
    nn_module=nn_sofm(dataset_minist,data_test)
    net=nn_module.net()
    upgrader = torch.optim.SGD(net.parameters(), lr=0.01)
    softmax1= softmax_classification.Softmax_Classfy()
    for epoch in range(10):
        for x,y in dataset_minist:
            y_pred=net(x)
            l=nn_module.loss(y,y_pred)
            l.backward()
            upgrader.step()
            upgrader.zero_grad()
        accuracy=0
        count=0
        for x, y in data_test:
            y_pred = net(x)
            y_sofm = softmax1.softmax(y_pred)
            y_pred_index = y_sofm.argmax(axis=1)
            Accum = y_pred_index.type(y.dtype) == y
            accuracy += float(Accum.mean(dtype=float))
            count += 1
        accuracy = accuracy / count
        print('Accuracy:%f' % accuracy)

imagedata= softmax_classification.ImageData()
train,test=imagedata.loadImage()
dataset_minist=imagedata.ImgdataLoader(train,10)
data_test=imagedata.ImgdataLoader(test,10)
opt='nn'
if opt=='mannul':
    mannul_sofm(dataset_minist,data_test)
elif opt=='nn':
    nn_sofm_train_test(dataset_minist,data_test)