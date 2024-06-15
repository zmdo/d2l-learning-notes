# 数据操作示例

import numpy as np
import torch
from torch.utils import data

# 首先，我们可以使用 arange 创建一个行向量 x。
# 这个行向量包含以0开始的前12个整数，它们默认创建为整数。
# 也可指定创建类型为浮点数。张量中的每个值都称为张量的 元素（element）。
# 例如，张量 x 中有 12 个元素。除非额外指定，新的张量将存储在内存中，并采用基于CPU的计算。
x = torch.arange(12)
print(x)

# 可以通过张量的shape属性来访问张量（沿每个轴的长度）的形状 。
print(x.shape)

# 如果只想知道张量中元素的总数，即形状的所有元素乘积，可以检查它的大小（size）。
# 因为这里在处理的是一个向量，所以它的shape与它的size相同。
print(x.numel())

y = torch.tensor([[ 1.0, 2.0, 3.2, 5.6],[3, 4.3, 5.0,6.6]])
print(y)

# 这里的 numel 意思是 number of elements 即元素的数量
print(y.numel())

print(y.reshape(4,2))

f12 = torch.arange(12,dtype = torch.float32)
print(f12.reshape(3,4))

# 创建一个 3 * 4 的连续数组
x = torch.arange(12).reshape(3,4)

# 输出第 3行第二列的值 ，即 9
print(x[2,1])