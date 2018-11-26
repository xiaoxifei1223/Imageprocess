# coding:utf-8

import pydicom
import os
import cv2
import numpy as np
if __name__ =="__main__":
    path = '/home/chenhao/device/Data/病例图像/'
    dcm = pydicom.read_file('/home/chenhao/device/Data/DicomImages/train2048/src/data/X15207198.dcm')
    savepath = '/home/chenhao/device/Data/病例图像_dcm/'
    lists = os.listdir(path)
    for file in lists:
        img = cv2.imread(path+file,cv2.IMREAD_UNCHANGED)
        if len(img.shape) > 2:
            img = img[0, :, :]
        if img.dtype == 'uint8':
            img = img.astype(np.uint16)
        row,col = img.shape
        dcm.__setattr__()
        dcm.__setattr__('Rows',row)
        dcm.__setattr__('Columns',col)
        dcm.PixelData = img
        file = file.split('.')[:-1][0] + '.dcm'
        dcm.save_as(savepath+file)
