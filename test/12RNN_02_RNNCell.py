# -*- coding: utf-8 -*-
"""
Created on  2021/9/19 18:39

@author: shengrihui
"""

# 使用RNNCell解决 hello 到 ohlol

import torch

input_size = 4
hidden_size = 4
batch_size = 1  # 因为只有一个样本

idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]  # 戴表idx2char的下标，所以是 hello
y_data = [3, 1, 2, 3, 2]  # ohlol

# 建立noe-hot向量
# 先建立one-hot字典，然后根据x_data的值作为下标选取one-hot
one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]
# print(x_one_hot)
# [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

"""
inputs=torch.tensor(x_one_hot)
print(inputs)
tensor([[0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
print(inputs.shape)
torch.Size([5, 4])
"""
# 现在的x_one_hot是(seq_len,input_size)的，
# 但我们还需要有batch_size这一维度，所以要view
inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
# print(inputs)
# tensor([[[0, 1, 0, 0]],
#         [[1, 0, 0, 0]],
#         [[0, 0, 1, 0]],
#         [[0, 0, 1, 0]],
#         [[0, 0, 0, 1]]])
# print(inputs.shape)
# torch.Size([5, 1, 4])

labels = torch.LongTensor(y_data).view(-1, 1)


# print(labels)
# print(labels.shape)
# tensor([[3],
#         [1],
#         [2],
#         [3],
#         [2]])
# torch.Size([5, 1])


class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.rnncell = torch.nn.RNNCell(input_size=self.input_size, hidden_size=self.hidden_size)
    
    def forward(self, input, hidden):
        return self.rnncell(input, hidden)
    
    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)


net = Model(input_size, hidden_size, batch_size)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

for epoch in range(15):
    loss = 0
    hidden = net.init_hidden()
    optimizer.zero_grad()
    print("Predicted string:", end='')
    for input, label in zip(inputs, labels):
        hidden = net(input, hidden)
        loss += criterion(hidden, label)
        _, idx = hidden.max(1)
        print(idx2char[idx], end='')
    loss.backward()
    optimizer.step()
    print(f"  epoch:{epoch + 1},loss:{loss.item()}")
