# coding: utf-8
'''
TODO: 判断是否需要对图像反色处理
author: chenhao
data:2018.11.13
'''

import numpy as np
import cv2
import os

def getCH(img,nbins=256):
    '''
    TODO: get img Cumulative distribution histogram
    :param img: ndarray input image
    :param nbins: integer histogram bins
    :return ch： ndarray result of Cumulative distribution histogram
    '''
    # get image histogram
    imgmax = img.max()
    imgmin = img.min()

    hist,bins = np.histogram(img.flatten(),nbins,[imgmin,imgmax])

    area = img.shape[0]*img.shape[1]
    # calculate cumulative histogram
    cdf = hist.cumsum()
    cdf_normalized = cdf /area   # get normalized cumulative distribution histogram

    return cdf_normalized, bins

def getMajorGrey(nbins,cd_normalized,th):
    '''
    TODO: get img majority object grey intensity distribution
    :param img:
    :param nbins:
    :param cd_normalized:
    :return:
    '''

    th_min = th
    th_max = 1.0 - th

    cd_normalized = np.array(cd_normalized)
    index_th_min = np.where(cd_normalized > th_min)[0][0]
    index_th_max = np.where(cd_normalized > th_max)[0][0]

    major_min = nbins[index_th_min].astype(np.uint16)
    major_max = nbins[index_th_max].astype(np.uint16)


    return major_max, major_min


def _judge_inverse(img,max):
    '''
    TODO: 根据四角判断是否局域主要成分的上位，如果是就进行反色，返回flag
    :param img:
    :param max:
    :param min:
    :return:
    '''
    row, col = img.shape
    block1 = img[0:10, 0:10]
    block2 = img[row-10:row, 0:10]
    block3 = img[0:10, col-10:col]
    block4 = img[row-10:row, col-10:col]

    m1 = np.mean(block1)
    m2 = np.mean(block2)
    m3 = np.mean(block3)
    m4 = np.mean(block4)

    Flag = False
    if m1 >= max:
        Flag = True
    if m2 >= max:
        Flag = True
    if m3 >= max:
        Flag = True
    if m4 >= max:
        Flag = True
    return Flag


def judge_inverse(img):
    cdf, bins = getCH(img, 256)
    grey_max, grey_min = getMajorGrey(bins, cdf, 0.05)
    flag = _judge_inverse(img, grey_max)
    return flag

if __name__ =="__main__":
    path = '/home/chenhao/device/Data/png_2/'
    lists = os.listdir(path)
    for file in lists:
        img = cv2.imread(os.path.join(path, file), cv2.IMREAD_UNCHANGED)
        cdf, bins = getCH(img, 256)
        grey_max, grey_min = getMajorGrey(bins, cdf, 0.05)
        flag = judge_inverse(img, grey_max)
        print("{} is judged, and result is {}". format(file, flag))
