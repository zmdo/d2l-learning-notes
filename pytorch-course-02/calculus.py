import numpy as np
# import matplotlib.pyplot as plt
import torch

print('--------[自动微分1]----------')
x = torch.arange(4.0)
# 等价于 x = torch.arange(4.0,requires_grad=True)
x.requires_grad_(True)
# 默认为 None
print(x.grad)
print('----------------------------')

print('--------[自动微分2]----------')
# 定义一个 y 函数
y = 2 * torch.dot(x, x)
# 输出正常点积的值
print(y)
# 这里使用反向传播函数计算梯度
print(y.backward())
print(x.grad)
# 其中 torch.dot(x,x) 的梯度应该为 2 * x
# 即 y = 2 * torch.dot(x,x) 的梯度应为 4 * x
print(x.grad == 4 * x)
print('----------------------------')

print('--------[自动微分3]----------')
# 默认情况下会累加 x 的梯度，这里我们需要将其清零
x.grad.zero_()
y = x.sum()
# 输出正常y的值
print(y)
# 这里使用反向传播函数计算梯度
y.backward()
print(x.grad)
# 其中 x.sum() 即每个元素之和，那么它的梯度的每个
# 维度上的分量就是 1
print(x.grad == torch.ones(4))
print('----------------------------')

print('-----[非标量变量的反向传播]-----')
# 默认情况下会累加 x 的梯度，这里我们需要将其清零
x.grad.zero_()
# 定义一个新的 y 函数
y = x * x
print(y)
# 等价于 y.backward(torch.ones(len(x)))
y.sum().backward()
print(x.grad)
print('----------------------------')

print('----------[分离计算]---------')
x.grad.zero_()
y = x * x
# 这里的 detach() 方法是将 y 跳过计算图
# 简而言之，就是不会再往前进行求导等运算了
u = y.detach()
z = u * x

z.sum().backward()
print(x.grad)
print(x.grad == u)

### 这里是正常情况下，不进行分离的情况：
x.grad.zero_()
u = y
z = u * x

z.sum().backward()
# 分量上应该为 3 * x^2
print(x.grad)
print('----------------------------')

print('----[Python控制流的梯度计算]---')

# 定义一个函数
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

# size = () 是随机的维度大小，不填默认为标量
a = torch.randn(size = (),requires_grad=True)
# a = torch.randn(size = (1,2),requires_grad=True)
print(a)

# 向后传播
d = f(a)
d.sum().backward()

print(a.grad == d/a)

print('----------------------------')