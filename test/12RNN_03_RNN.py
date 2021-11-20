# -*- coding: utf-8 -*-
"""
Created on  2021/9/20 20:45
    
@author: shengrihui
"""

# 使用RNN

import torch

batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2
num_layers = 5

cell = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

inputs = torch.randn(seq_len, batch_size, input_size)
hidden = torch.zeros(num_layers, batch_size, hidden_size)
print('inputs', inputs)
print('inputs.shape', inputs.shape)
print('hidden', hidden)
print('hidden.shape', hidden.shape)

output, hidden = cell(inputs, hidden)
print('out', output)
print('out.shape', output.shape)
# out.shape (seq_len,batch_size,hidden_size)
print('hidden', hidden)
print('hidden.shape', hidden.shape)
# hidden.shape (num_layers,batch_size,hidden_size)

a = output[-1][0]
b = hidden[-1][0]
print(a, b, a == b)
