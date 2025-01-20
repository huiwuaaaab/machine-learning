import torch


def corr(input,k,stride=1,padding=0):
    x=input.shape[0]
    y=input.shape[1]
    knel=torch.randn(size=(k,k))
    out_i=(x+2*padding-k)/stride+1
    out_j=(y+2*padding-k)/stride+1
    output=torch.zeros((out_i,out_j))
    for i in range(0,out_i,stride):
        for j in range(0,out_j,stride):
            output[i,j]=torch.sum(knel[i:i+k,j:j+k]*x[i:i+k,j:j+k])