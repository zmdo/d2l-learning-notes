import torch
from torch import nn

import model_train
import torchvision
from torch.utils import data
from torchvision import transforms

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

# 定义网络
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784,256),
    nn.ReLU(),
    nn.Linear(256,10)
)

# 权重初始化
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,std= 0.01)
net.apply(init_weights)

# 交叉熵损失函数
loss = nn.CrossEntropyLoss()

# 获取数据
train_iter, test_iter = dataload()

# 定义超参数
num_epochs, lr = 20, 0.1
updater = torch.optim.SGD(net.parameters(), lr=lr)

# 模型训练
model_train.train_ch3(net,train_iter,test_iter,loss,num_epochs,updater)

