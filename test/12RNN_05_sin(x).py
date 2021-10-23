# -*- coding: utf-8 -*-
"""
Created on  2021/10/23 9:51

@author: shengrihui
"""

# 用RNN预测sin(X)

import numpy as np
import torch
import torch.nn.functional  as F
import matplotlib.pyplot  as plt
import os

input_size = 1
batch_size = 1
hidden_size = 5
num_layers = 2
lr = 0.002
# 输入前40个点，预测后10个
pred_len = 10
data_len = 50
input_len = 40


class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers, pred_len, input_len):
        super(Net, self).__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_len = pred_len
        self.rnn = torch.nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.linear = torch.nn.Linear(hidden_size * input_len, pred_len)
    
    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = torch.tanh(out)
        out = out.view(self.batch_size, -1)
        out = self.linear(out)
        out = out.view(self.pred_len, self.batch_size, -1)
        return out, hidden
    
    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)


net = Net(input_size=input_size, hidden_size=hidden_size, batch_size=batch_size, num_layers=num_layers,
          pred_len=pred_len, input_len=input_len)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)


def train():
    if os.path.exists('12RNN_05_sin(x).pth'):
        net.load_state_dict(torch.load('12RNN_05_sin(x).pth'))
    loss_ = []
    for epoch in range(6000):
        start = np.random.randint(0, 3, 1)
        data_x = np.linspace(start, start + data_len, data_len)
        data_y = np.sin(data_x)
        inputs = torch.tensor(data_x[:input_len]).float().view(input_len, batch_size, 1)
        labels = torch.tensor(data_y[input_len:]).float().view(pred_len, batch_size, 1)
        
        optimizer.zero_grad()
        out, hidden = net(inputs, net.init_hidden())
        # out=torch.sigmoid(out)
        # print(out.shape,labels.shape)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        print(f'epoch:{epoch}  loss:{loss.item()} ')
        loss_.append((loss.item()))
        
        if epoch == 5999:
            for i in range(pred_len):
                print(round(abs(out[i].item() - labels[i].item()), 4))
        # exit(0)
    torch.save(net.state_dict(), '12RNN_05_sin(x).pth')
    plt.plot(range(6000), loss_)
    plt.show()


# pred
def pred():
    if os.path.exists('12RNN_05_sin(x).pth'):
        net.load_state_dict(torch.load('12RNN_05_sin(x).pth'))
    else:
        train()
    start = 100
    data_x = np.linspace(start, start + data_len, data_len)
    data_y = np.sin(data_x)
    plt.plot(data_x, data_y, color='blue')
    inputs = torch.tensor(data_x[:input_len]).float().view(input_len, batch_size, input_size)
    out, hidden = net(inputs, net.init_hidden())
    out = out.detach().numpy().reshape(-1)
    plt.plot(data_x[input_len:], out, color='red')
    plt.show()


if __name__ == '__main__':
    # train()
    pred()
