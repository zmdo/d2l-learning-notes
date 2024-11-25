import torch
from torch import nn
import torchvision
from torch.utils import data
from torchvision import transforms
import model_train

threads = 0
batch_size = 256

threads = 0
batch_size = 256

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

dropout1, dropout2 = 0.2,0.5

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784,256),
    nn.ReLU(),

    nn.Dropout(dropout1),
    nn.Linear(256,256),
    nn.ReLU(),

    nn.Dropout(dropout2),
    nn.Linear(256, 10)
)

def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std = 0.01)

net.apply(init_weight)

num_epochs, lr = 10, 0.5

train_iter, test_iter = dataload()
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr = lr)

model_train.train_ch3(net, train_iter,test_iter, loss, num_epochs, trainer)