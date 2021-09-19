# -*- coding: utf-8 -*-
"""
Created on Mon May 31 20:39:48 2021

@author: 11200
"""

#梯度下降算法
#以03线性模型的为基础做梯度下降

import numpy as np
import matplotlib.pyplot as plt

x_data=[1,2,3]
y_data=[2,4,6]
cost_list=[]
#任意先选一个w
w=1.0
#训练次数
Epoch=100
def forward(x):
    return x*w

#绘制cost曲线
def show_plot():
    plt.plot(list(range(Epoch)),cost_list)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()


#梯度下降
def GradientDescent():
    
    #同一个w的时候，对所有x，y计算
    #返回的是mse
    def cost(xs,ys):
        cost=0
        for x,y in zip(xs,ys):
            y_pred=forward(x)
            loss=(y_pred-y)**2
            cost+=loss
        return cost/len(xs)
    
    #计算不同w下的梯度
    #算法由解析式推的
    def gradient(xs,ys):
        grad=0
        for x,y in zip(xs,ys):
           grad+=2*x*(x*w-y)
        return grad/len(xs)
    
    
    global w,cost_list
    w=-555
    cost_list=[]
    #训练Epoch次
    for epoch in range(Epoch):
        #计算mse主要是为了打印日志
        cost_val=cost(x_data,y_data)
        grad_val=gradient(x_data,y_data)
        #更新w  w=w-α*grad α学习率，不宜过大

        cost_list.append(cost_val)
        w -= 0.01*grad_val
        #当梯度为0的时候，w不在更新，但实际上一般到不了0
        #日志
        print("Epoch {:2d}:   w={}   cost={}".format(epoch,w,cost_val))
        
    show_plot()

#随机梯度下降
#对一个样本进行梯度下降
def StochaticGradient():
    def loss(x,y):
        return (forward(x)-y)**2
    
    def gradient(x,y):
        return 2*x*(x*w-y) 
    
    global w,cost_list
    w=100
    cost_list=[]
    for epoch in range(Epoch):
        for x,y in zip(x_data,y_data):
            grad=gradient(x,y)
            w -= 0.01*grad
            l=loss(x,y)
        cost_list.append(l)
        print("Epoch {:2d}:   w={}   cost={}".format(epoch,w,l))
    show_plot()
    
if __name__=='__main__':
    GradientDescent()
    
    StochaticGradient()
    
    """
    局部最优，全局最优
    鞍点
    随机梯度下降可以跳过鞍点，可以找到较优的结果性能高，
    但没两个样本之间有联系（下一个的w由前一个w来），不能并行，时间复杂度高
    
    梯度下降没两次计算之间没有联系，可以进行并行，时间复杂度低，
    但不容易找到较优结果，，性能低
    
    需要性能高，时间复杂度高
    折中办法，Batch
    批量梯度下降，批量随机梯度下降
    应该叫Mini-Batch
    
    可以通过 指数加权均值 解决cost函数震荡
    第三集视频 29分左右
    """
    
    
    
    
