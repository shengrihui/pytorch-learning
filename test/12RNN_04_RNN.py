# -*- coding: utf-8 -*-
"""
Created on  2021/9/20 21:25

@author: shengrihui
"""

# 使用RNN解决 hello 到 ohlol
import torch

input_size = 4
hidden_size = 4
batch_size = 1
num_layers = 1
seq_len = 5


class model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size):
        super(model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.rnn = torch.nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
    
    def forward(self, input, hidden):
        return self.rnn(input, hidden)
    
    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)


net = model(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_size=batch_size)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]
y_data = [3, 1, 2, 3, 2]

one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]

inputs = torch.Tensor(x_one_hot).view(seq_len, batch_size, input_size)
labels = torch.LongTensor(y_data).view(seq_len, -1)


def out2str(out):
    _, idx = torch.max(out, 2)
    s = [idx2char[i[0]] for i in idx]
    return ''.join(s)


# train
for epoch in range(50):
    optimizer.zero_grad()
    outputs, hidden = net(inputs, net.init_hidden())
    loss = 0.0
    for i in range(seq_len):
        loss += criterion(outputs[i], labels[i])
    # loss=criterion(outputs,labels)
    loss.backward()
    optimizer.step()
    ret = out2str(outputs)
    print(f'epoch:{epoch}  loss:{loss.item():.6f}  {ret}')
    # exit(0)
