# test2
实现前向传播
import numpy as np

# 定义权重 (w1~w8)
W = np.array([
    [0.2, -0.4],   # x1 -> h1, x1 -> h2 的权重
    [0.5,  0.6],   # x2 -> h1, x2 -> h2 的权重
    [0.1, -0.5],   # h1  -> o1, h1 -> o2 的权重
    [0.3,  0.8]    # h2  -> o1, h2 -> o2 的权重
])

# 定义输入 x (形状为 (2,1) 或者 (1,2)，根据矩阵乘法维度来)
x = np.array([[0.5], [0.3]])

# 若考虑偏置，也可以定义 b_h（隐藏层偏置）、b_o（输出层偏置）
# 这里假设题目没给偏置，先省略；若有偏置，需额外定义 b_h = np.array([...]), b_o = np.array([...])
# 定义 Sigmoid 激活函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 输入层 -> 隐藏层：矩阵乘法 + 激活
# W 的前两行对应 x->h1, x->h2 的权重；x 是 (2,1)，所以 W[:2, :] 是 (2,2)？不，更简单的是：
# 把 x 转成 (2,1)，然后和 W 的前两行（对应 x->h1, x->h2）做乘法
net_h = np.dot(W[:2, :], x)  # W[:2, :] 形状 (2,2)，x 形状 (2,1) → net_h 形状 (2,1)
h = sigmoid(net_h)           # 隐藏层输出，形状 (2,1)
# 隐藏层 -> 输出层
net_o = np.dot(W[2:, :], h)  # W[2:, :] 形状 (2,2)，h 形状 (2,1) → net_o 形状 (2,1)
o = sigmoid(net_o)           # 输出层输出，形状 (2,1)

print("NumPy 实现的输出 o：\n", o)
import torch

# 定义权重 (和 NumPy 形状一致)
W = torch.tensor([
    [0.2, -0.4],
    [0.5,  0.6],
    [0.1, -0.5],
    [0.3,  0.8]
], dtype=torch.float32)

# 定义输入 x
x = torch.tensor([[0.5], [0.3]], dtype=torch.float32)

# 若有偏置，也可以定义 b_h = torch.tensor([...]), b_o = torch.tensor([...])
# 定义 Sigmoid 激活函数
def sigmoid(z):
    return 1 / (1 + torch.exp(-z))

# 输入层 -> 隐藏层
net_h = torch.matmul(W[:2, :], x)  # 矩阵乘法
h = sigmoid(net_h)                 # 隐藏层输出
# 隐藏层 -> 输出层
net_o = torch.matmul(W[2:, :], h)
o = sigmoid(net_o)

print("PyTorch 实现的输出 o：\n", o)
