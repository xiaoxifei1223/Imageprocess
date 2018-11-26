# coding:utf-8
'''
TODO: 封装对target的修饰算法
author: chenhao
data:2018.11.09
'''
import numpy as np
def _crop(img1,img2,mask):
    # img1 是被抠图的图像，img2是将抠出的内容进行填充的图像，mask是指导抠图的图像
    index = np.where(mask == 1)
    index_tuple = list(map())