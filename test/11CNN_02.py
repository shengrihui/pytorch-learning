# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 13:49:25 2021

@author: 11200
"""

import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((1307,), (0.3081,))
])

# train
train_dataset = datasets.MNIST(root='MNIST',
                               train=True,
                               download=False,
                               transform=transform)
train_loader = DataLoader(dataset=train_dataset,
                          shuffle=True,
                          batch_size=batch_size)

# test
train_dataset = datasets.MNIST(root='MNIST',
                               train=False,
                               download=False,
                               transform=transform)
test_loader = DataLoader(dataset=train_dataset,
                         shuffle=False,
                         batch_size=batch_size)
