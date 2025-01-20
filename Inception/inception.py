import torch.nn as nn
import torch

class inception(nn.Module):
    def __init__(self,input_channel,out_channel,channel1,channel2,channel3,channel4):
        super().__init__()
        self.x1=nn.Conv2d(input_channel,out_channel,1)
        self.x2=nn.Sequential(nn.Conv2d(input_channel,channel1[0],1),nn.Conv2d(channel1[0],channel1[1],3,padding=1))
        self.x3=nn.Sequential(nn.Conv2d(input_channel,channel2[0],1),nn.Conv2d(channel2[0],channel2[1],5,padding=2))
        self.x4=nn.Sequential(nn.MaxPool2d(3,padding=1),nn.Conv2d(input_channel,out_channel,1))
        self.f=nn.ReLU()
    def forward(self,x):
        x1=self.f(self.x1(x))
        x2=self.f(self.x2(x))
        x3=self.f(self.x3(x))
        x4= self.f(self.x4(x))
        x=torch.cat((x1,x2,x3,x4),dim=1)
        return x
