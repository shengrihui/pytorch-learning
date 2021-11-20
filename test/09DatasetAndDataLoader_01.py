# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 21:19:57 2021

@author: 11200
"""

# 使用Dataset 和 DataLoader

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        # print(xy.shape) #(759, 9)
        # 获得数据的行数
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 4)
        self.linear2 = torch.nn.Linear(4, 2)
        self.linear3 = torch.nn.Linear(2, 1)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


if __name__ == '__main__':
    filepath = "diabetes.csv.gz"
    
    dataset = DiabetesDataset(filepath)
    
    train_loader = DataLoader(dataset=dataset,
                              batch_size=32,
                              shuffle=True,
                              num_workers=2)
    
    model = Model()
    criterion = torch.nn.BCELoss(size_average=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    Epoch = 100
    for epoch in range(Epoch):
        cost = 0.0
        for idx, data in enumerate(train_loader):
            inputs, targets = data
            
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cost += loss.item()
        
        print(epoch, cost)
