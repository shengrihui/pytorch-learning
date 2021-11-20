# -*- coding: utf-8 -*-
"""
Created on  2021/10/24 16:23

@author: shengrihui
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

input_size = 2
input_len = 80
out_size = 2
out_len = 20
data_len = 100
hidden_size = 10
batch_size = 1
num_layers = 2
lr = 0.002
pt_path = '12RNN_LSTM_sin(x)cos(s).pt'


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, out_size, input_len, batch_size, out_len):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.out_size = out_size
        self.input_len = input_len
        self.out_len = out_len
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=False,
        )
        self.linear = nn.Sequential(
            nn.Linear(self.input_len * self.hidden_size,
                      self.hidden_size),
            nn.Sigmoid(),
            nn.Linear(self.hidden_size, self.out_len * self.out_size),
        )
    
    def forward(self, x, hidden):
        out, (h, c) = self.lstm(x, hidden)
        out = torch.tanh(out)
        out = out.view(self.batch_size, -1)
        out = self.linear(out)
        out = out.view(self.out_len, self.batch_size, -1)
        return out
    
    def init_h_c(self):
        h = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        c = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        return h, c


net = Net(input_size=input_size,
          hidden_size=hidden_size,
          batch_size=batch_size,
          num_layers=num_layers,
          out_size=out_size,
          input_len=input_len,
          out_len=out_len)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
criterion.to(device)


def train():
    if os.path.exists(pt_path):
        net.load_state_dict(torch.load(pt_path))
    loss_ = []
    tbar = tqdm(range(6000))
    for epoch in tbar:
        start = np.random.randint(0, 100, 1)
        data_x = np.linspace(start, start + data_len, data_len)
        data_y1 = np.sin(data_x).reshape(-1, 1)
        data_y2 = np.cos(data_x).reshape(-1, 1)
        data_y = np.hstack((data_y1, data_y2))
        inputs = torch.tensor(data_y[:input_len]).float().view(input_len, batch_size, input_size)
        labels = torch.tensor(data_y[input_len:]).float().view(out_len, batch_size, out_size)
        
        inputs, labels = inputs.to(device), labels.to(device)
        h, c = net.init_h_c()
        h, c = h.to(device), c.to(device)
        print(inputs.shape, h.shape)
        exit(0)
        optimizer.zero_grad()
        out = net(inputs, (h, c))
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        # print(f'epoch:{epoch}  loss:{loss.item()} ')
        tbar.set_description('Epoch:%d  loss:%.10f ' % (epoch, loss.item()))
        
        loss_.append((loss.item()))
    
    torch.save(net.state_dict(), pt_path)
    plt.plot(range(6000), loss_)
    plt.show()


def pred(start=20):
    if os.path.exists(pt_path):
        net.load_state_dict(torch.load(pt_path))
    else:
        train()
    net.to(torch.device('cpu'))
    data_x = np.linspace(start, start + data_len, data_len)
    data_y1 = np.sin(data_x)
    data_y2 = np.cos(data_x)
    plt.scatter(data_x[input_len:], data_y1[input_len:], color='blue')
    plt.scatter(data_x[input_len:], data_y2[input_len:], color='blue')
    data_y = np.hstack((data_y1.reshape(-1, 1), data_y1.reshape(-1, 1)))
    inputs = torch.tensor(data_y[:input_len]).float().view(input_len, batch_size, input_size)
    
    out = net(inputs, net.init_h_c())
    out = out.detach().numpy().reshape(-1, 2)
    # print(out)
    plt.scatter(data_x[input_len:], out[:, 0], color='red')
    plt.scatter(data_x[input_len:], out[:, 1], color='red')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    train()
    # pred(500)
