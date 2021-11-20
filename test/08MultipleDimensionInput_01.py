# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 19:20:50 2021

@author: 11200
"""
# 多维输入

import numpy as np
import torch
import matplotlib.pyplot as plt

# numpy可以直接加载gz文件，
# delimiter分隔符
# 用32位是因为在N卡上够用了
xy = np.loadtxt('diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
# [-1]是要形成矩阵
y_data = torch.from_numpy(xy[:, [-1]])


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        # 这是一个对象，激活函数，一个层
        # s要大写
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()
criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_list = []

Epoch = 100
for epoch in range(Epoch):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    
    loss_list.append(loss.item())
    print(epoch, loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.figure(figsize=(16, 9), dpi=80)
plt.plot(range(100), loss_list)
plt.show()
