# 绘制图表学习
import matplotlib.pyplot as plt
import torch

def f(x):
    return 3 * x ** 2 - 4 * x

# 求函数极限的公式
# f : 函数
# x : 当前值
# h : 增量
def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h

h = 0.1
x = 1
xs = []
yd = []
for i in range(5):
    xs += [x + h]
    yd += [numerical_lim(f,x,h)]
    h *= 0.1
    x = x + h

plt.plot(xs,yd)
plt.draw()
plt.show()
