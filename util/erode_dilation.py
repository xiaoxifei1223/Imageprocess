# coding:utf-8
# author: chenhao

import cv2
import numpy as np


def erode(img, size):
    kernel = np.ones((size, size), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    return img


def dilate(img, size):
    kernel = np.ones((size, size), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    return img


def enlarge_mask(img, size):
    # size取值最好是设为偶数
    row,col = img.shape
    img_new = cv2.resize(img, (int(col + size), int(row + size)))
    temp = img_new[int(size/2):row + int(size/2), int(size/2):col + int(size/2)]
    return temp



