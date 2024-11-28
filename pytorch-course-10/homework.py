import torch
from torch import nn

def corr2d(X, K):
    h,w = K.shape
    Y = torch.zeros( (X.shape[0] - (h - 1),X.shape[1] - (w - 1)) )
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h,j:j +w] * K).sum()
    return Y

class MyConv2d(nn.Module):

    def __init__(self,kernel,bias):
        super().__init__()
        self.weights = kernel
        self.bias = 0

    def forward(self, X):
        return corr2d(X, self.weights) + self.bias

X = torch.ones((6, 8))
X[:, 2:6] = 0
print(X)


conv2d = MyConv2d( kernel=torch.tensor([[0., 1.],[2., 3.]]), bias= 0 )
print(conv2d.forward(X))
