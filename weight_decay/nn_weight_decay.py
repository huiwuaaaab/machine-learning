import collections

import matplotlib.pyplot as plt
import torch.nn
import torch



def CreateData(x_num,feature_num,a,b,nois=True):
    size_X=(x_num,feature_num)
    x=torch.normal(0,0.1,size=size_X)
    if nois:
        noise=torch.normal(0,0.2,size=size_X)
    else:
        noise=0
    y=a*x**2+b*x+noise
    return x,y

class Net(object):
    def __init__(self,feature,output):
        self.feature=feature
        self.output=output

    def net(self):
        self.nnnet=torch.nn.Sequential(torch.nn.Linear(self.feature,self.output))
        return self.nnnet

    def loss(self):
        l=torch.nn.MSELoss(reduction='mean')
        return l

    def upgrator(self,lr,weightdecay=True):
        if weightdecay:
            return torch.optim.SGD([{'params':self.nnnet[0].weight,'weight_decay':0.01},{'params':self.nnnet[0].bias}],lr=lr)
        else:
            return torch.optim.SGD(self.nnnet[0].parameters(),lr=lr)

def plot_loss(loss_train,loss_test,epoch):
    dev_x=epoch
    dev_y_t=loss_train
    dev_y_te=loss_test
    plt.plot(dev_x,dev_y_t)
    plt.plot(dev_x,dev_y_te)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train loss','test loss'])
    plt.show()

if __name__=='__main__':
    feature,label=CreateData(20,200,0.1,1.24)
    test_f,test_l=CreateData(50,200,0.1,1.24)
    Net_class=Net(200,1)
    net=Net_class.net()
    l=Net_class.loss()
    optimizer=Net_class.upgrator(0.01,weightdecay=False)
    epochs=30
    loss_dic=collections.defaultdict(list)
    for epoch in range(epochs):
        label_pred=net(feature)
        loss=l(label_pred,label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        with torch.no_grad():
            label_pred=net(feature)
            train_loss=l(label_pred,label)
            loss_dic['loss_train'].append(train_loss)
            label_pred = net(test_f)
            test_loss = l(label_pred, test_l)
            loss_dic['loss_test'].append(test_loss)
    plot_loss(**loss_dic,epoch=[x for x in range(epochs)])
