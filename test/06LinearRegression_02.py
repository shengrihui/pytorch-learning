# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 21:14:28 2021

@author: 11200
"""

import torch
import torch.optim as op
import matplotlib.pyplot as plt


class myModel(torch.nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
    
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = myModel()

criterion = torch.nn.MSELoss(size_average=False)

optimizer_list = [op.Adagrad, op.Adam, op.Adamax, op.ASGD, op.LBFGS, op.RMSprop, op.Rprop, op.SGD]
optimizer_list_str = ['Adagrad', 'Adam', 'Adamax', 'ASGD', 'LBFGS', 'RMSprop', 'Rprop', 'SGD']

plt.figure(figsize=(16, 9), dpi=80)


def Train(op):
    optimizer = optimizer_list[op](model.parameters(), lr=0.01)
    
    x_data = torch.tensor([[1.0], [2.0], [3.0]])
    y_data = torch.tensor([[2.0], [4.0], [6.0]])
    w_list = []
    b_list = []
    loss_list = []
    
    Epoch = 200
    for epoch in range(Epoch):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        
        w = model.linear.weight.item()
        w_list.append(w)
        b = model.linear.bias.item()
        b_list.append(b)
        l = loss.data.item()
        loss_list.append(l)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # print(f'process:{epoch} w:{w:.2f} b:{b:.2f} loss:{l:.2f}')
    
    x_ = range(Epoch)
    # plt.plot(x_,w_list,label='w')
    # plt.plot(x_,b_list,label='b')
    plt.plot(x_, loss_list, label=optimizer_list_str[op])
    plt.legend()
    plt.grid(alpha=0.6)


for i in range(0, len(optimizer_list)):
    print(optimizer_list_str[i])
    print(i)
    if i == 4:
        continue
    Train(i)
plt.show()
