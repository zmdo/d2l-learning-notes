import torch
from torch import nn
import torchvision
from torch.utils import data
from torchvision import transforms
import model_train

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

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1

    if dropout == 1:
        return torch.zeros_like(x)
    if dropout == 0:
        return X

    mask = (torch.rand(X.shape) > dropout).float()

    # 保持期望一致
    # E(h') = 0xp + h/(1-p)*(1-p) = h
    return mask * X / (1.0 - dropout)

dropout1, dropout2 = 0.2, 0.5

class Net(nn.Module):
    def __init__(self,num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training = True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1,self.num_inputs))))

        if self.training == True:
            H1 = dropout_layer(H1,dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            H2 = dropout_layer(H2,dropout2)
        out = self.lin3(H2)
        return out

num_inputs = 784
num_hiddens1 = 256
num_hiddens2 = 64
num_outputs = 10

# 构建网络
net = Net(num_inputs,num_outputs,num_hiddens1,num_hiddens2)

# 学习轮数、学习率
num_epochs, lr = 10, 0.5

# 损失函数
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = dataload()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
model_train.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)