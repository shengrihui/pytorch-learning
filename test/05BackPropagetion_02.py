# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 18:52:29 2021

@author: 11200
"""

import torch
import matplotlib.pyplot as plt


def forward(x):
    return w1 * x * x + w2 * x + b


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w1 = torch.tensor([10.0])
w1.requires_grad = True
w2 = torch.tensor([-5.5])
w2.requires_grad = True
b = torch.tensor([3.0])
b.requires_grad = True

Epoch = 5000
w1_list = []
w2_list = []
b_list = []
cost_list = []

for epoch in range(Epoch):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        
        w1.data -= 0.01 * w1.grad.data
        w2.data -= 0.01 * w2.grad.data
        b.data -= 0.01 * b.grad.data
    
    print(f'process:{epoch} w1:{w1.data.item():.2f} w2:{w2.data.item():.2f} \
          b:{b.data.item():.2f} loss:{l.data.item():.2f} ')
    w1.grad.zero_()
    w2.grad.zero_()
    b.grad.zero_()
    
    cost_list.append(l.data.item())
    w1_list.append(w1.data.item())
    w2_list.append(w2.data.item())
    b_list.append(b.data.item())

plt.figure(figsize=(160, 90), dpi=80)

# plt.plot(range(Epoch),cost_list)
plt.plot(range(Epoch), w1_list, label='w1')
plt.plot(range(Epoch), w2_list, label='w2')
plt.plot(range(Epoch), b_list, label='b')
plt.legend()
plt.grid(alpha=0.6)
plt.show()
