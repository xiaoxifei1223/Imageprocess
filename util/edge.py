# coding:utf-8
# TODO: 生成肋骨边缘的轮廓
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import skimage.morphology as morphology
from util.erode_dilation import dilate

def find_edge(img,min,max):
    edge = cv2.Canny(img, min, max)
    return edge


if __name__ =="__main__":
    path = r'G:\Data\edge_mask'
    save_path = r'G:\Data\ribedge_mask'
    lists = os.listdir(path)
    count = 0
    for file in lists:
        print(file)
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        mean_img = np.mean(img)
        img[img < mean_img] = 0
        img[img > mean_img] = 255
        img = img.astype(np.uint8)
        edge = find_edge(img, 10, 200)
        edge = dilate(edge, 1)
        edge = cv2.resize(edge, (2048,2048))
        img_save_path = os.path.join(save_path, file)
        cv2.imwrite(img_save_path, edge)
        count+=1
    print(count)
