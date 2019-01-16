# coding:utf-8
# TODO:生成原图减去软组织的图像

import cv2
import pydicom

def inverse(img):
    max_val = img.max()