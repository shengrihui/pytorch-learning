# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 21:09:42 2021

@author: 11200
"""

# 用pytorch实现线性回归模型

import torch


class LinearModal(torch.nn.Module):
    
    def __init__(self):
        # 继承父类的__init__()
        super(LinearModal, self).__init__()
        '''
        Linear是继承Module的一个类
        参数是输入和输出的数据的维度，一样
        bias是否要算上偏移量b，默认True
        Linear也是继承Module的一个类
        '''
        self.linear = torch.nn.Linear(1, 1)
    
    # 建立模型的时候，必须要重写__init__()和forward()
    # 对象的__call__就是调用forward()
    def forward(self, x):
        # 直接调用Linear类的方法，
        # 就可以不用管他里面具体是怎么实现的了
        y_pred = self.linear(x)
        return y_pred


# 实例化模型
model = LinearModal()

'''
MSELoss也是继承Module的一个计算MSE损失函数的类
参数：size_average是否求均值，因为效果一样，所以False
但，在最后的时候，样本数可能不够（之前都是100个，最后只有32个），
这时候可能就需要求均值（其实影响也不大）
reduce：是否降维
criterion-准则
'''
criterion = torch.nn.MSELoss(size_average=False)

'''
构造优化器（梯度下降）
参数：
model.parameters()：model是自己的model，
parameters()是父类里的方法，可以找到model里所有的权重
lr 学习lv
pytorch支持在模型的不同的部分使用不同的学习率
'''
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 数据要做成一列三行的形式，
# 列代表维度，一行是一个样本
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])

# 训练过程
Epoch = 1000
for epoch in range(Epoch):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(f'{model.linear.weight.item():.4f},{model.linear.bias.item():.4f}')
    
    # print(epoch,loss.data.item())
    # 打印loss的时候，会自动调用对象的__str__()
    
    # 梯度手动清零
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 更新权重
    optimizer.step()
