import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20,256)
        self.out = nn.Linear(256,10)
        self.relu = nn.ReLU()
    def forward(self,X):
        return self.out(F.relu(self.hidden(X)))

net = MLP()
X = torch.randn(size = [1,20])
print(X.device)
print(X)
print(net(X))

# 实现序列
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。_module的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X

X = torch.rand(size = [1,20])
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
print(f'MySequential {net(X)}')

# 固定隐藏层
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 隐藏层权重
        self.rand_weight = torch.rand((20,20),requires_grad=False)
        self.linear = nn.Linear(20,20)

    def forward(self,X):
        # 输入层
        X = self.linear(X)

        # 隐藏层
        X = F.relu(torch.mm(X,self.rand_weight) + 1)

        # 输出层
        X = self.linear(X)

        return X.sum()
