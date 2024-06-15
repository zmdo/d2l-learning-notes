import torch

A = torch.arange(12,dtype=torch.float32).reshape(3,4)

# 张量克隆
B = A.clone()

print('---------[张量转置]----------')
# 张量转置
print(A.T)
print('----------------------------')

print('-----[shape相等的张量相加]----')
# 张量加法
print(A + B)
# 张量加上标量，其为每一个元素都加标量
print(A + 1)
print('----------------------------')

print('----[shape不相等的张量相加]----')
# 注意：张量加法之不同大小的张量加法
# 程序会自动扩展较小的张量，使之相加
M = torch.arange(16,dtype=torch.float32).reshape(2,8)
N = torch.arange(8,dtype=torch.float32).reshape(1,8)
print(M + N)
print('----------------------------')

print('--------[Hadmard乘法]--------')
# 张量按元素乘法（Hadamard 乘法）
print(A * B)
# 张量乘标量
print(A * 3)
print('----------------------------')


print('--------[张量元素之和]--------')
# 求所有元素之和
print(A.sum())
# 求沿着哪一个轴求和
print(A.shape)
print(A.sum(axis=0))
print(A.sum(axis=1))
print(A.sum(axis=[0,1]))
# 非降维求和,有时在调用函数来计算总和或均值时保持轴数不变会很有用。
print(A.sum(axis=1, keepdims=True))
# 非降维求和，生成的结果中会单独增加一个元素来表示求和的值
print(A.cumsum(axis=1))
print('----------------------------')

print('---------[张量平均值]---------')
# 利用平均值公式求平均值
print(A.sum()/A.numel())
print(A.sum(axis=0)/A.shape[0])

# 利用 mean() 方法求平均值
print(A.mean())
print(A.mean(axis=0))
print('----------------------------')

print('----------[向量点积]---------')
# 生成两个向量
X = torch.arange(12,dtype=torch.float32)
Y = torch.arange(12,dtype=torch.float32)
# 向量点积,按元素相乘然后求和
print(torch.dot(X,Y))
# 这里的可以用张量乘法然后求和表示
print(torch.sum(X*Y))
print('----------------------------')

print('-------[矩阵-向量点积]--------')
# 生成一个 5*4 的矩阵
M54 = torch.arange(20).reshape(5,4)
# 生成一个 4 维的向量
V4 = torch.arange(4)
# 用矩阵与向量做点积，这里的 mv 即 matrix 与 vector 的缩写
print(torch.mv(M54,V4))
print('----------------------------')

print('---------[矩阵点积]----------')
# 生成一个 5*4 的矩阵
M54 = torch.arange(20).reshape(5,4)
# 生成一个 4*5 的矩阵
M45 = torch.arange(20).reshape(4,5)
# 用矩阵与向量做点积，这里的 mm 即 matrix 与 matrix 的缩写
print(torch.mm(M54,M45))
print('----------------------------')

print('---------[向量范数]----------')
# 生成一个二维的向量
X2 = torch.tensor([3.0,4.0],dtype=torch.float32)
# 范数的英文名为 norm ，这里提供的 norm() 方法是指的 L2 范数
print(X2.norm())
# L1 范数的求法
print(X2.abs().sum())
print('----------------------------')

print('---------[矩阵范数]----------')
# 矩阵的 Frobenius 范数类似于向量的 L2 范数
# 求的是每个元素的平方之和再开平方根
print(A.norm())
print('----------------------------')
