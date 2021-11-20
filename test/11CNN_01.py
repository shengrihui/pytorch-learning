# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 16:00:36 2021

@author: 11200
"""

import torch

# 输入和输出的通道数，
# 也就是卷积核的通道数，和和卷积核的数量
in_channels, out_channels = 5, 10

# 输入的宽和高
width, height = 100, 100

# 卷积核的大小，一般是正方形奇数，但也可以是长方形，偶数
kernel_size = 3

batch_size = 1


def test1():
    inputs = torch.randn(batch_size,
                         in_channels,
                         width,
                         height)
    conv_layer = torch.nn.Conv2d(in_channels,
                                 out_channels,
                                 kernel_size=kernel_size)
    
    outputs = conv_layer(inputs)
    
    print(inputs.shape)
    print(outputs.shape)
    print(conv_layer.weight.shape)


def test2():
    a, b = 64, 128
    inputs = [i for i in range(a * b)]
    inputs = torch.Tensor(inputs).view(1, 1, a, b)  # (Batch,channel,h,w)
    kernel = torch.Tensor([i for i in range(1, 10)]).view(1, 1, 3, 3)  # (O,I,H,W)
    conv_layer = torch.nn.Conv2d(1,
                                 1,
                                 kernel_size=3,
                                 stride=2,
                                 bias=False)
    conv_layer.weight.data = kernel.data
    
    outputs = conv_layer(inputs)
    # output=torch.nn.MaxPool2d(2)
    print(inputs.shape)
    print(outputs.shape)
    print(conv_layer.weight.shape)


test2()
