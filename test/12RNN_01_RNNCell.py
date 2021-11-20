# -*- coding: utf-8 -*-
"""
Created on  2021/9/19 18:07

@author: shengrihui
"""

# 使用RNNC、ell

import torch

# 设置参数
batch_size = 1
# 序列长度，相当于有多少个x输入，一个单词当中有多少个字母
# 每次循环拿出此时刻所有的输入
seq_len = 3
# 输入的维度，one-hot向量的维度=seq_len*input_size
input_size = 4
hidden_size = 2

# 需要指定输入和隐藏的维度
cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

# （seq_len,batch_size,input_size)
# 循环的时候遍历seqlen，取出(batch_size,input_size)
dataset = torch.randn(seq_len, batch_size, input_size)
# print(dataset)
# tensor([[[-0.5446,  1.4580, -0.2285, -0.9541]],
#         [[-0.9974,  0.0500, -1.4568,  0.4851]],
#         [[-0.3668, -1.7975, -2.3452, -0.1734]]])

# c初始隐藏层设为全零，(batch_size,hidden_size)
hidden = torch.zeros(batch_size, hidden_size)
# print(hidden)
# # tensor([[0., 0.]])

for idx, input in enumerate(dataset):
    print('=' * 20, idx, '=' * 20)
    print('input :', input)
    print('input size:', input.shape)
    
    # input=(batch_size,input_size)
    # -> (batch_size,hidden_size)
    # +
    # hidden=(batch_size,hidden_size)
    # ->(batch_size,hidden_size)
    # ->
    # output(hidden)=(batch_size,hidden_size)
    hidden = cell(input, hidden)
    print('output/hidden :', hidden)
    print('putput/hidden size:', hidden.shape)
    # ==================== 0 ====================
    # input : tensor([[-1.0720, -1.0083,  1.0325,  0.7219]])
    # input size: torch.Size([1, 4])
    # output/hidden : tensor([[-0.6823, -0.8587]], grad_fn=<TanhBackward>)
    # putput/hidden size: torch.Size([1, 2])
    # ==================== 1 ====================
    # input : tensor([[-1.6405,  0.0827, -0.5267,  1.1621]])
    # input size: torch.Size([1, 4])
    # output/hidden : tensor([[ 0.6336, -0.8969]], grad_fn=<TanhBackward>)
    # putput/hidden size: torch.Size([1, 2])
    # ==================== 2 ====================
    # input : tensor([[-0.8617,  1.0453, -1.4648,  2.7871]])
    # input size: torch.Size([1, 4])
    # output/hidden : tensor([[-0.2993, -0.9475]], grad_fn=<TanhBackward>)
    # putput/hidden size: torch.Size([1, 2])
