import torch.nn as nn
import torch.nn.functional as F
import torch
from weight_decay.nn_weight_decay import CreateData,plot_loss
import collections

class model(nn.Module):
    def __init__(self,input,lin1,output,prob):
        super().__init__()
        self.linear1=nn.Linear(input,lin1)
        self.linear2=nn.Linear(lin1,output)
        self.dropout=nn.Dropout(p=prob)

    def forward(self,x):
        x1=F.relu(self.linear1(x))
        x1_d=self.dropout(x1)
        output=F.relu(self.linear2(x1_d))
        return output

if __name__=='__main__':
    feature,label=CreateData(20,200,0.1,1.24)
    test_f,test_l=CreateData(50,200,0.1,1.24)
    l=nn.MSELoss(reduction='mean')
    net=model(200,100,1,0.5)
    epochs=20
    loss_dic = collections.defaultdict(list)
    optimizer=torch.optim.SGD(net.parameters(),0.1)
    for epoch in range(epochs):
        net.train()
        y_pred=net(feature)
        loss = l(y_pred, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        with torch.no_grad():
            net.eval()
            y_pred = net(feature)
            train_loss = l(y_pred, label)
            loss_dic['loss_train'].append(train_loss)
            label_pred = net(test_f)
            test_loss = l(label_pred, test_l)
            loss_dic['loss_test'].append(test_loss)
    plot_loss(**loss_dic,epoch=[x for x in range(epochs)])