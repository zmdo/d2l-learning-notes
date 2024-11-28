import torch
from torch import nn

K = torch.tensor([ [[0., 1.],[2., 3.]], [[4., 5.],[6., 7.]]])
X = torch.stack((K, K +1, K + 2),0)

print(X.shape)
print(X)


conv2d = nn.Conv2d(
    in_channels=1,
    out_channels=1,
    kernel_size=(2, 2),
    dilation=(2,2),
    padding=0,
    padding_mode='zeros',
    stride=1,
    bias=False
)

conv2d.weight.data = torch.tensor([[1., 2.],[3., 4.]]).reshape((1,1,2,2))
print(conv2d.weight.data)

X = torch.tensor([
    [1., 6., 1.],
    [6., 5., 6.],
    [1., 6., 1.]
]).reshape((1, 1, 3, 3))

print(X)
print(conv2d(X))
