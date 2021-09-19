# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 11:09:25 2021

@author: 11200
"""

import torch
import matplotlib.pyplot as plt


def forward(x):
    #因为w是Tensor，所以x会自动转成Tensor
    return x*w

def loss(x,y):
    y_pred=forward(x)
    return (y-y_pred)**2

Epoch=100
x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]

w=torch.tensor([1.0])
#float 才能设requires_grad
w.requires_grad=True

w_list=[]
loss_list=[]

for epoch in range(Epoch):
    #随机梯度下降
    for x,y in zip(x_data,y_data):
        #每次只需要计算loss，
        #这个过程就是建立计算图的过程
        l=loss(x,y)
        l.backward()
        '''
        反向传播后才有下面这些
        print(w.grad)  #tensor([-2.])
        print(w.grad.data)  #tensor([-2.])
        print(w.grad.item())  #-2.0
        '''
        #print(w,w.grad)
        w.data = w.data- 0.01 * w.grad.data
        #print(w,w.grad)
    #将w的grad清零（因木星的需要决定是否清零）
    #不然  有问题
    #print()
    w.grad.zero_()
    loss_list.append(l.item())
    print(f'progress:{epoch} w:{w.item():.2f} loss:{l.item():.2f}')
    
plt.plot(range(Epoch),loss_list)
plt.show()