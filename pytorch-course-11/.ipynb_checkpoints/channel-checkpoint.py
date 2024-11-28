import torch
from torch import nn

K = torch.tensor([ [[0., 1.],[2., 3.]], [[4., 5.],[6., 7.]]])
X = torch.stack((K, K +1, K + 2),0)

print(X.shape)
print(X)