# coding: utf-8
'''
TODO: 图像剪切后粘贴,图像的crop操作集成
author: chenhao
email: haochen9212@outlook.com
data: 2018.11.12
'''
import numpy as np
import os
import cv2
import pydicom
import numpngw

# def crop(img1, img2, mask):
#     # 将img1按照mask将img2的内容填进去
#     row, col = img1.shape
#     img_new = np.zeros((row, col), dtype = np.uint16)
#     for i in range(row):
#         for ii in range(col):
#             if mask[i, ii] !=1:
#                 img_new[i, ii] = img1[i, ii]
#             else:
#                 img_new[i, ii] = img2[i, ii]
#     return img_new



def crop_aligned(img1,img2):
    '''
    TODO: 将img1和img2结合为一张图并进行保存
    :param img1:
    :param img2:
    :return:
    '''
    new_img = np.zeros((512, 1024), dtype=np.uint16)
    new_img[0:512, 0:512] = img1
    new_img[0:512, 512:1024] = img2
    return new_img



if __name__ =="__main__":
    path1 = '/home/chenhao/device/Data/DicomImages/train512/trainA/'
    path2 = '/home/chenhao/device/Data/DicomImages/train512/trainB/'
    lists = os.listdir(path1)
    save_path = '/home/chenhao/device/Data/DicomImages/train512/train/'
    for file in lists:
        file1_path = os.path.join(path1, file)
        file2_path = os.path.join(path2, file)
        img1 = cv2.imread(file1_path, cv2.IMREAD_UNCHANGED)
        img2 = cv2.imread(file2_path, cv2.IMREAD_UNCHANGED)
        new_img = crop_aligned(img1, img2)
        save_img_path = os.path.join(save_path, file)
        numpngw.write_png(save_img_path, new_img)