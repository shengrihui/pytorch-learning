# -*- coding: utf-8 -*-
"""
Created on Mon May 31 16:32:45 2021

@author: 11200
"""

#线性模型

import numpy as np
import matplotlib.pyplot as plt

#前馈函数，以y=x*w为模型
def forward(x):
    return x*w

#损失函数
def loss(x,y):
    y_pred=forward(x)
    return (y_pred-y)**2


#准备训练数据集
x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]

#w列表
w_list=[]
#mse-平均平方误差
mse_list=[]

#穷举法在可能的范围内找最合适的w
for w in np.arange(0.0,4.0,0.2):
    print("w=",w)
    loss_sum=0
    for x,y in zip(x_data,y_data):
        y_pred=forward(x)
        loss_val=loss(x,y)
        loss_sum+=loss_val
        print('\t',x,y,y_pred,loss_val)
    mse=loss_sum/len(x_data)
    print('MSE=',mse,'\n')
    w_list.append(w)
    mse_list.append(mse)
    
plt.plot(w_list,mse_list)
plt.ylabel('mes')
plt.xlabel('w')
plt.show()

