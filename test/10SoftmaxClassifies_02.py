# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 21:08:35 2021

@author: 11200
"""

# MNIST

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


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)


model = Model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        
        outputs = model(inputs)
        loss = criterion(outputs, target)
        # 0print(loss)
        running_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 300 == 299:
            print('[%d %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(correct / total * 100)


for epoch in range(100):
    train(epoch)
    test()
