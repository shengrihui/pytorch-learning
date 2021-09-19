# -*- coding: utf-8 -*-
"""
Created on Sat May 29 20:08:30 2021

@author: 11200
"""
#torch 读取数据为dataset

import cv2
import json
import numpy as np

import torch
#torch 里 常用工具 数据
from torch.utils.data import Dataset

import glob

class my_dataset(Dataset):
    
    def __init__(self,img_path,img_labels):
        self.img_path=img_path
        self.img_labels=img_labels
        
    def __getitem__(self, index):
        img=cv2.imread(self.img_path[index])
        lbl=self.img_labels[index]
        return img,lbl
    def __len__(self):
        return len(self.img_path)
        
#获取路径
train_path = glob.glob('../data/mchar_train/mchar_train/*.png')
train_json=json.load(open('../data/mchar_train.json'))
train_labels=[ train_json[x]['label'] for x in train_json]

#print(train_path)
#pprint.pprint(train_labels)


a=my_dataset(train_path, train_labels)
l=a[2][1]
print(a[2][1])
cv2.imshow('a',a[2][0])

#设置lbl长度不超过5个
lbl = np.array(l, dtype=np.int)
lbl = list(lbl) + (5 - len(lbl)) * [10]
#tensor类型
x=torch.from_numpy(np.array(lbl[:5]))
print(x)