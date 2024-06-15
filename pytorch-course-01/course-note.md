# 第一课 课程笔记

## 数据操作

### 创建张量

创建张量 (tensor) , 需要引入 `torch` 进行操作：

1. 使用数组矩阵直接创建

```python
import torch

# 这里创建一个 1 * 3 的张量
x = torch.tensor([1,2,3])
```

2. 使用 `arange` 进行创建

```python
import torch

# 这里创建一个 0 -> 11 的张量
x = torch.arange(12)

# 指定类型
x = torch.arange(12,dtype=torch.float32)
```

### 张量数据类型

在这里，他会创建一个 [1,2,3] 的一维向量，这里的类型都是整型，所以可以使用以下两种方式对数据进行处理，让其变为浮点型：

1. 直接在元素上使用浮点数的表示方式，值得注意的是，浮点数默认是 `float64` 类型的：

```python
import torch
x = torch.tensor([1,2,3.0])    
```

2. 在生成张量时指定类型 `dtype` ：

```python
import torch    
x = torch.tensor([1,2,3],dtype=torch.float64)
```
一般情况下， `float64` 类型的数据用的较少，因为计算量较大，计算较慢，所以一般情况下，采用的是 `float32` 类型的数据。

### 张量的数据结构

查看张量的数据结构有两种方式，分别是 `shape` 和 `numel`。

* `shape` 指的是张量的“维度”，其是一个<font color="red">**向量**</font>。
* `numel` 的意思是 "number of elements" 即**元素的数量**，其是一个<font color="red">**常量**</font>。

下面是一个示例：

```python
import torch
# 创建一个张量 x ，其为一个三行四列的矩阵：
#      ┌ 1  2  3  4 ┐
# x =  │ 5  6  7  8 │
#      └ 9 10 11 12 ┘
x = torch.tensor([[1,2,3,4],[5,6,7,8],[9,10,11,12]])

print(x.shape)   # 返回的是 torch.Size([3, 4]) ，即该张量的维度为 3 * 4
print(x.numel()) # 返回的是 12 ，即该张量有 12 个元素
```

### 访问元素

1. 使用数组的定位方式访问元素

```python
import torch

# 创建一个 3 * 4 的连续数组
x = torch.arange(12).reshape(3,4)

# 输出第 2 行第 1 列的值 ，即 tensor(9)
# 注意这里的 9 不是标量，而是一个 1*1 的张量
# 可以通过 int(x[2,1]) 或 float(x[2,1]) 的方式将数据转换为标量
print(x[2,1])
```

2. 使用切片的方式访问元素

```python
import torch

# 创建一个 3 * 4 的连续数组
x = torch.arange(16).reshape(4,4)

# 获取一行数据
print(x[1,:])

# 获取一列数据
print(x[:,1])

## 注意：
## 1. 下面描述的行数和列数都是从0开始计数的
## 2. 从 m 到 n 结果包含 m 不包含 n

# 子区域1： 获取一片矩形子区域
# 即从 第一行到第三行（不包含第三行） 的第一列到最后一列
print(x[1:3,1:])

# 子区域2： 跳跃取数 
# 从第0行到最后一行每隔三行元素取一次值（包含第0行）
# 从第0列到最后一列每隔两列取一次值（包含第0列）
print(x[::3,::2])
```

根据切片的取值方式，可以将冒号与数字间的关系总结为“**左闭右开三步数**”

