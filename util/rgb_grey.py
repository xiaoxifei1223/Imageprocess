# coding:utf-8
'''
TODO: 将3通道rgb转为单通道grey
author: chenhao
data: 2018.11.09
email: haochen9212@outlook.com
'''
import numpngw
import os
import cv2

def rgb_png(img):
    if len(img.shape) ==3:
        img_new = img[:, :, 0]
    else:
        img_new = img
    return img_new

if __name__ =="__main__":
    path = '/home/chenhao/device/Data/Dong_Dong/'
    save_path = '/home/chenhao/device/Data/Dong_png/'
    lists = os.listdir(path)
    for file in lists:
        img_path = os.path.join(path,file)
        img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
        img = rgb_png(img)
        img_save_path = os.path.join(save_path,file)
        numpngw.write_png(img_save_path,img)
