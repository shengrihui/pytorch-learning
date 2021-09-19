# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 17:18:46 2021

@author: 11200
"""
import torch

import torchvision
train_set=torchvision.datasets.MNIST(root='MNIST',train=True,download=True)

test_set=torchvision.datasets.MNIST(root='MNIST',train=False,download=True)


image=torch.load('MNIST/MNIST/processed/training.pt')
print(image)
print(len(image))
print(image[0].shape)