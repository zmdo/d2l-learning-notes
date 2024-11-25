# 多层感知机

import torch
from torch import nn
import torchvision
from torch.utils import data
from torchvision import transforms

import model_train

# 一共多少批次
batch_size = 256
threads = 0

# 参数

def dataload():
    # 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
    # 并除以255使得所有像素的数值均在0～1之间
    trans = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=threads),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=threads))


def rule(X):
    # 获得一个根 X 形状相同，但值都为0的矩阵
    a = torch.zeros_like(X)
    return torch.max(X, a)

def net(X):
    X = X.reshape((-1,num_inputs))
    H = rule(X @ W1 + b1)
    return (H @ W2 + b2)

# 交叉熵损失函数
loss = nn.CrossEntropyLoss()

# 获取数据
train_iter, test_iter = dataload()

# 定义网络
num_inputs, num_outputs, num_hiddens = 784,10,256
W1 = nn.Parameter(torch.randn(
    num_inputs,num_hiddens,requires_grad=True
))
b1 = nn.Parameter(torch.zeros(
    num_hiddens,requires_grad=True
))

W2 = nn.Parameter(torch.randn(
    num_hiddens,num_outputs,requires_grad=True
))
b2 = nn.Parameter(torch.zeros(
    num_outputs,requires_grad=True
))

params = [W1, b1, W2, b2]

# 定义超参数
num_epochs, lr = 20, 0.1
updater = torch.optim.SGD(params, lr=lr)

# 模型训练
model_train.train_ch3(net,train_iter,test_iter,loss,num_epochs,updater)
