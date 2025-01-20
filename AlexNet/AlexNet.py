import torch.nn as nn
import torchvision
from LeNet.lenet import gpu_train
from torchvision.transforms import transforms
import torch.utils.data


class Alexnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(3, 96, 11, 4),
                            nn.ReLU(),
                            nn.MaxPool2d(3, 2),
                            nn.Conv2d(96, 256, 5, 1, 2),
                            nn.ReLU(),
                            nn.MaxPool2d(3, 2),
                            nn.Conv2d(256, 384, 3, 1, 1),
                            nn.ReLU(),
                            nn.Conv2d(384, 384, 3, 1, 1),
                            nn.ReLU(),
                            nn.Conv2d(384, 256, 3, 1, 1),
                            nn.ReLU(),
                            nn.MaxPool2d(3, 2))

        self.fc=nn.Sequential(nn.Linear(256*6*6, 4096),
                            nn.ReLU(),
                            nn.Dropout(),
                            nn.Linear(4096, 4096),
                            nn.ReLU(),
                            nn.Dropout(),
                            nn.Linear(4096, 10),
                            nn.Softmax(1))
    def forward(self,x):
        x1=self.conv(x)
        x2=self.fc(x1.view(x.shape[0],-1))
        return x2

def ImgdataLoader(dataset,batch_size):
    batch_size=batch_size
    dataset=torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)
    return dataset


if __name__=='__main__':
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=Alexnet().to(device=device)
    transform=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(227),transforms.ToTensor()])
    train=torchvision.datasets.CIFAR10(root='./data',train=True,transform=transform)
    test=torchvision.datasets.CIFAR10(root='./data',train=False,transform=transform)
    trainloader=ImgdataLoader(train,128)
    valoader=ImgdataLoader(test,50)
    gpu_train(model,trainloader,valoader,10,0.01,momentum=0.9,weightdecay=0.0005)