# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 17:40:47 2021

@author: 11200
"""

# 逻辑斯蒂回归 二分类

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class LogisticModel(torch.nn.Module):
    def __init__(self):
        super(LogisticModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # 仍需要线性模型
    
    def forward(self, x):
        # 在torch里，sigmoid就是Logistic函数
        y_pred = F.sigmoid(self.linear(x))
        return y_pred


model = LogisticModel()

# BCE损失函数
# 是否选择求均值，影响学习率的选择
criterion = torch.nn.BCELoss(size_average=False)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

Epoch = 10
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[0.0], [0.0], [1.0]])
loss_list = []

for epoch in range(Epoch):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(y_pred, y_data)
    loss_list.append(loss.item())
    print(epoch, round(loss.item(), 2))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

'''
plt.figure(figsize=(16,9),dpi=80)
plt.plot(range(Epoch),loss_list)
plt.show()
'''

x = np.linspace(0, 10, 200)
x_t = torch.tensor(x, dtype=torch.float).view((200, 1))
y_t = model(x_t)
y = y_t.data.numpy()
plt.plot(x, y)
plt.plot([0, 10], [0.5, 0.5], c='r')
plt.grid()

plt.show()
