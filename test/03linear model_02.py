# -*- coding: utf-8 -*-
"""
Created on Mon May 31 17:28:52 2021

@author: 11200
"""
#线性模型 二维

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def forward(x,w,b):
    return x*w+b

def loss(y,y_pred,w,b):
    return (y_pred-y)**2

x_data=[1,2,3,4,5,6]
#y=2x+0.5
y_data=[2.5,4.4,6.3,7.5,10.5,12.6]
#y_data=[x*2+0.5 for x in x_data]
len_x=len(x_data)

w_list=[]
b_list=[]
mse_list=[]

w_m,b_m,mse_m=0,0,100
for w in np.arange(0,4,0.2):
    print('w=',w)
    mse_l=[]
    for b in np.arange(0,2.0,0.1):
        print('\tb=',b)
        loss_sum=0
        for x,y in zip(x_data,y_data):
            y_pred=forward(x, w, b)
            loss_val=loss(y,y_pred,w,b)
            loss_sum+=loss_val
            print('\t\t',x,y,y_pred,loss_val)
        mse=loss_sum/len_x
        if mse<mse_m:
            w_m,b_m,mse_m=w,b,mse
        
        mse_l.append(mse)
        print('\tMSE=',mse)
        if b not in b_list:
            b_list.append(b)
    w_list.append(w)
    mse_list.append(mse_l)
    print('\n')

print(w_m,b_m,mse_m)

figure=plt.figure()
plt.plot(x_data,y_data)
y=[x*w_m+b_m for x in x_data ]
plt.plot(x_data,y)

def _3d():
    W,B=np.meshgrid(w_list,b_list)
    
    figure=plt.figure()
    ax=Axes3D(figure)
    MSE=np.array(mse_list)
    
    ax.plot_surface(W,B,MSE)

plt.show()