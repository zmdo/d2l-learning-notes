import torch
from torch import nn
import model_train

leNet = nn.Sequential(
    nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,padding=2),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2,stride=2),
    nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2,stride=2),
    nn.Flatten(),
    nn.Linear(16*5*5,120),
    nn.Sigmoid(),
    nn.Linear(120,84),
    nn.Sigmoid(),
    nn.Linear(84,10)
)

X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in leNet:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)