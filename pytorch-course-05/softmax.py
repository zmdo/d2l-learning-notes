import torch


# X = torch.tensor([[1.0, 2.0, 3.0],[4.0, 5.0, 6.0]])
# print(X.sum(0, keepdim=True), X.sum(1,keepdim=True))

# 定义 softmax 函数
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition


# 定义网络
def net(X):
    return softmax(torch.matmul(X.reshape(-1, W.shape[0])), W + b)


y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
print(y_hat[[0, 1], y])
