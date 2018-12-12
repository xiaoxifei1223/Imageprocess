# coding:utf-8
# author: chenhao
# data:2018.12.12
# TODO: 设计去雾算法用于目标边缘的增强处理，以达到消除雾感的效果
import numpy as np
import cv2

def haze_remover(img,soft_max):
    # 初始化参数
    param_balance = -1.0
    param_percentage = 0.95
    param_win = 11

    # 检验输入
    soft_max = soft_max.astype(np.float)
    if soft_max < 1:
        print("soft_max is out of bound")

    row,col = img.shape
    nbo = img.copy()
    img_pro = img.copy()
    img_prom = cv2.medianBlur(img_pro, param_win)
    img_pros = abs(img_pro - img_prom)
    img_prosm = cv2.medianBlur(img_pros, param_win)
    b = img_prom - img_prosm

    v = np.zeros((row,col))
    for i in range(row):
        for j in range(col):
            if b[i,j] < img_pro[i,j]:
                v[i,j] = b[i,j]
            else:
                v[i,j] = img_pro[i,j]

            if v[i,j] < 0:
                v[i,j] = 0
    v = param_percentage*v

    factor = 1.0/(1.0 - v)

    r = np.zeros((row, col))
    r = (img - v)

