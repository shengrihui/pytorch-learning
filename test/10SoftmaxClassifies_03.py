# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 20:03:40 2021

@author: 11200
"""


#MNIST
#与10_02用不一样的加载数据的方法

import torch
from torchvision import datasets,transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset

batch_size=64
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((1307,), (0.3081,))
    ])


class Data(Dataset):
    def __init__(self,path):
        self.data,self.labels=torch.load(path)
        self.len=self.data.shape[0]
    def __getitem__(self,index):
        return self.data[index],self.labels[index]
    def __len__(self):
        return self.len
#train
train_dataset=Data('MNIST/MNIST/processed/training.pt')
train_loader=DataLoader(dataset=train_dataset,
                        shuffle=True,
                        batch_size=batch_size)

#test
test_dataset=Data('MNIST/MNIST/processed/test.pt')
test_loader=DataLoader(dataset=train_dataset,
                        shuffle=False,
                        batch_size=batch_size)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1=torch.nn.Linear(784,512)
        self.l2=torch.nn.Linear(512,256)
        self.l3=torch.nn.Linear(256,128)
        self.l4=torch.nn.Linear(128,64)
        self.l5=torch.nn.Linear(64,10)
    
    def forward(self,x):
        x=x.view(-1,784)
        x=F.relu(self.l1(x))
        x=F.relu(self.l2(x))
        x=F.relu(self.l3(x))
        x=F.relu(self.l4(x))
        return self.l5(x)

model=Model()
criterion=torch.nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

def train(epoch):
    running_loss=0.0
    for batch_idx,data in enumerate(train_loader,0):
        inputs,target=data
        
        #元数据是int但后面需要float
        inputs=torch.tensor(inputs,dtype=torch.float32)
        outputs=model(inputs)
        loss=criterion(outputs,target)
        #0print(loss)
        running_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 300 ==299:
            print('[%d %5d] loss: %.3f' %(epoch+1,batch_idx+1,running_loss/300))
            running_loss=0.0
       
def test():
    correct=0
    total=0
    #不需要计算梯度
    with torch.no_grad():
        for data in test_loader:
            inputs,labels=data
            
            inputs=torch.tensor(inputs,dtype=torch.float32)
            outputs=model(inputs)
            #找输出的最大值的下标，max返回两个值 值和雄安表
            _,predicted=torch.max(outputs.data,dim=1)
            print(torch.max(outputs.data,dim=1))
            #labels是一个n*1的tensor,size返回元组(n,1)
            total+=labels.size(0)
            print(labels.size,labels.shape)
            #张量之间的比较运算
            correct+=(predicted==labels).sum().item()
        print(correct/total*100)


for epoch in range(10):
    train(epoch)
    test()
