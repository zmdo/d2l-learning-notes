import torch
from torch import nn

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

class ListSequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.module_list = []
        for idx,item in enumerate(args):
            self.module_list.append(item)

    def forward(self,X):
        for block in self.module_list:
            X = block(X)
        return X

X = torch.rand(size = [1,20])

in_linear = nn.Linear(20, 256)
rule = nn.ReLU()
out_linear = nn.Linear(256, 10)

net1 = MySequential(
    in_linear,
    rule,
    out_linear
)

net2 = ListSequential(
    in_linear,
    rule,
    out_linear
)

print(f'MySequential {net1(X)}')
print(f'ListSequential {net2(X)}')