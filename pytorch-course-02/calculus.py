import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 3 * x ** 2 - 4 * x

# 求函数极限的公式
# f : 函数
# x : 当前值
# h : 增量
def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h

h = 0.1
for i in range(5):
    print(f'h={h:.5f},numerical limit={numerical_lim(f,1,h):.5f}')
    h *= 0.1

print('---------[自动微分]----------')

print('----------------------------')