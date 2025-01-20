import os
import torchvision
import torch.utils.data
from torchvision.transforms import transforms

class ImageData(object):
    def __init__(self):
        pass

    def loadImage(self,root='./data'):
        self.trans=transforms.ToTensor()
        self.root=root
        if not os.path.exists(self.root):
            os.mkdir(self.root)
            os.mkdir(os.path.join(self.root,'train'))
            os.mkdir(os.path.join(self.root,'test'))
        self.imgdata_train=torchvision.datasets.MNIST(root='./data/train',transform=self.trans,train=True,download=True)
        self.imgdata_test=torchvision.datasets.MNIST(root='./data/test',train=False,transform=self.trans,download=True)
        return self.imgdata_train,self.imgdata_test

    def ImgdataLoader(self,dataset,batch_size):
        self.batch_size=batch_size
        dataset=torch.utils.data.DataLoader(dataset,batch_size=self.batch_size,shuffle=True)
        return dataset

class Softmax_Classfy(object):
    def __init__(self):
        pass

    def softmax(self,pred):
        self.pred=pred
        self.pred_sum=torch.sum(torch.exp(self.pred),1)
        return torch.exp(self.pred)/self.pred_sum

    def loss(self,pred,true):
        y_pred=pred[:,true]
        l=-torch.log(y_pred).mean()
        return l

class Linear_regression(object):
    def __init__(self):
        pass

    def linear(self,w,b,feature):
        self.w=w
        self.b=b
        self.x=feature
        return torch.matmul(self.x.reshape(-1,self.w.shape[0]),self.w)+self.b

    def upgrade(self,lr,w,b,w_grad,b_grad):
        with torch.no_grad():
            w-=lr*w_grad
            b-=lr*b_grad
        return w,b

class Accumulate():
    def __init__(self,n):
        self.data=[0.0]*n

    def add(self,x):
        self.data=[a+float(b) for a,b in zip(self.data,x)]