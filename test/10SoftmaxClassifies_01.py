# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 20:02:18 2021

@author: 11200
"""

# softmax函数
# NLL损失
# 处理多酚类问题

import numpy as  np
import torch


# softmax函数，NLL损失具体实现
def test1():
    # y是
    # z是三个softmax之前的结果
    y = np.array([1, 0, 0])
    z = np.array([0.2, 0.1, -0.1])
    
    y_pred = np.exp(z) / np.exp(z).sum()
    loss = (-y * np.log(y_pred)).sum()
    print(loss)


# 用torch实现
def test2():
    y = torch.LongTensor([0])
    z = torch.tensor([[0.2, 0.1, -0.1]])
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(z, y)
    print(loss)


def test3():
    Y = torch.LongTensor([2, 0, 1])
    criterion = torch.nn.CrossEntropyLoss()
    
    y_pred1 = torch.tensor([[0.1, 0.2, 0.9],
                            [1.2, 0.3, 0.1],
                            [0.1, 0.8, 0.3]])
    y_pred2 = torch.tensor([[0.5, 0.1, -0.2],
                            [0.0, 1.5, 0.6],
                            [1.1, -0.2, 0.0]])
    l1 = criterion(y_pred1, Y)
    l2 = criterion(y_pred2, Y)
    print(l1, l2)


test1()
test2()
test3()
