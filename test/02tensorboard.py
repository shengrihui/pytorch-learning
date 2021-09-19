# -*- coding: utf-8 -*-
"""
Created on Sun May 30 10:29:02 2021

@author: 11200
"""
#tensorboard的使用 可是跨

import cv2
import numpy as np
from torch.utils.tensorboard import SummaryWriter

#实例化
writer=SummaryWriter("02tensorboard_logs")

"""
Args:
    tag (string): Data identifier
    img_tensor (torch.Tensor, numpy.array, or string/blobname): Image data
    global_step (int): Global step value to record
img格式有要求，且是要CHW形式，
如果不是，用dataformats=’HWC'
"""
for i in range(10):
    img=cv2.imread('E:/CS/cv/tianchi/data/mchar_train/mchar_train/00000{}.png'.format(i))
    writer.add_image('my_img',img,i,dataformats='HWC')


'''
add_scalar
Args:
    tag (string): Data identifier
    scalar_value (float or string/blobname): Value to save
    global_step (int): Global step value to record
参数简单理解：标题，y轴，x轴

如果连续两次在同一个tag里画东西，会发生拟合
解决方法是删掉文件
'''

for i in range(100,130):
    writer.add_scalar('y=x2',i*3,i)

#dd_scalars(
# r = 5
# for i in range(100，200):
#     writer.add_scalars('run_14h', {'xsinx':i*np.sin(i/r),
#                                     'xcosx':i*np.cos(i/r),
#                                     'tanx': np.tan(i/r)}, i)
"""
如何查看
在cmd里先进入到本文件加，以及该环境当中
命令：tensorboard --logdir=02tensorboard_log(文件夹名） --port=6008(设定端口)，默认6006)
"""

writer.close()