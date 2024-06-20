import torch
# distributions 分布
# multinomial 多项式
from torch.distributions import multinomial

# 这里模拟的是一个投骰子的的概率，即每个面上的概率为 1/6
fair_probs = torch.ones(6,dtype=torch.float32)/6
# 这里是采样过程，在投骰子的概率分布上采10次样， sample 采样
result = multinomial.Multinomial(10,fair_probs).sample()
print(result)

# 进行 500 批采样，每次投 10 次
count = 10
batch = 500
result = multinomial.Multinomial(count,fair_probs).sample((batch,))

seq = result.cumsum(dim = 0)
count_matrix = seq.sum(dim=1, keepdims=True)
print(seq / count_matrix)