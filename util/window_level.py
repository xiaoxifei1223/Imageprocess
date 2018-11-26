# coding:utf-8
'''
TODO: reset gray-image window and level
author: chenhao
email: haochen9212@outlook.com
data: 2018.10.22
'''

import numpy as np
import os
import cv2
import pydicom
cdf_config={
'Nbins':256,
'Quantile':0.02,
'Slop_factor':0.2,
'Search_radius':5
}

def getCH(img,nbins=256,th=0.001):
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
    # maybe cutdown histogram
    area = img.shape[0]*img.shape[1]
    if hist[0]/(img.shape[0]*img.shape[1]) > th:
        area = area-hist[0]
        hist[0] = 0

    # calculate cumulative histogram
    cdf = hist.cumsum()
    cdf_normalized = cdf /area   # get normalized cumulative distribution histogram

    return cdf_normalized

def getInverseCH(q,cdf):
    '''
    TODO: get CH inverse Cumulative histogram function value
    :param q: float64 quantile
    :param cdf: ndarray float64 cumulative histogram
    :return value: float64 inverse function value
    '''
    length = len(cdf)
    index = int(q*length)
    return cdf[index]

def slop(Hmax,Hmin,factor):
    '''
    TODO: calculate slop
    :param Hmax: float constant
    :param Hmin: float constant
    :param factor: float config parameter
    :return value: float slop ability
    '''
    return (Hmax - Hmin - factor)/(Hmax - Hmin)

def getMax_left(x,width,hist,F_p,Hmax,Hmin,q):
    '''
    TODO: optimizer function
    :param q:
    :param width:
    :return:
    '''
    ls = []
    x = int(x*len(hist))
    for i in range(-width,width+1,1):
        ls.append(adjust_fun_left((x+i)/len(hist),hist,F_p,Hmax,Hmin,q))

    index = ls.index(max(ls))

    return index,max(ls)

def getMax_right(x,width,hist,F_p,Hmax,Hmin,q):
    '''
    TODO: optimizer function
    :param q:
    :param width:
    :return:
    '''
    ls = []
    x = int(x*len(hist))
    for i in range(-width,width+1,1):
        ls.append(adjust_fun_right((x+i)/len(hist),hist,F_p,Hmax,Hmin,q))

    index = ls.index(max(ls))

    return index,max(ls)

def adjust_fun_left(x,hist,F_p,Hmax,Hmin,q):
    '''
    TODO: calculate function
    :param x: integer index
    :param F_p: float constant
    :param Hmax: float constant
    :param Hmin: float constant
    :param q: float slop ability constant
    :return value: float64
    '''
    up = F_p - getInverseCH(x,hist)
    down = (Hmax - Hmin)*q**(x-Hmin)
    value = up/down
    return value

def adjust_fun_right(x,hist,F_anti_p,Hmax,Hmin,q):
    '''
    TODO: calculate function
    :param x: integer index
    :param F_p: float constant
    :param Hmax: float constant
    :param Hmin: float constant
    :param q: float slop ability constant
    :return value: float64
    '''
    up = getInverseCH(x,hist) - F_anti_p
    down = (Hmax - Hmin)*q**(Hmax-x)
    value = up/down
    return value



def cal_window_level(img,config):
    '''
    TODO: calculate gray image`s window and level
    :param img: ndarray image
    :param config: dict parameters
    :return window : integer image window
    :return level : integer image level
    '''
    # initialize
    Hmax = img.max()
    Hmin = img.min()
    nbins = config['Nbins']
    slop_factor = config['Slop_factor']
    quantile = config['Quantile']
    anti_quantile = 1.0-quantile
    r = config['Search_radius']
    q = slop(Hmax,Hmin,slop_factor)

    # calculate cdf
    cdf = getCH(img,nbins)

    # get quantile
    F_p = getInverseCH(quantile,cdf)
    F_anti_p = getInverseCH(anti_quantile,cdf)

    # get left window
    index,_ = getMax_left(quantile,r,cdf,F_p,Hmax,Hmin,q)

    winleft = int(quantile*(Hmax-Hmin)) + index

    # get right window
    index,_ = getMax_right(anti_quantile,r,cdf,F_anti_p,Hmax,Hmin,q)
    winright = int(anti_quantile*(Hmax-Hmin)) + index

    # get window and level
    window = winright - winleft
    level = int((winright+winleft)/2)

    return window,level


def get_W_L_simple(img,left,right):
    cdf = getCH(img,256)
    index = []
    flag = True
    i=0
    while flag:
        i+=1
        if cdf[i] > left:
            flag = False
    index.append(i)
    flag = True
    i=0
    while flag:
        i+=1
        if cdf[i] > right:
            flag = False
    index.append(i)

    winleft = int((index[0]/256)*(img.max()-img.min()))
    winright = int((index[1]/256)*(img.max()-img.min()))
    window = winright-winleft
    level = int((winleft+winright)/2)
    return window,level


def get_W_L_simple_distribution(img):
    imgmax = img.max()
    imgmin = img.min()

    Nbins = 256

    cdf_normalized = getCH(img,Nbins)
    flag = True
    i=0
    while flag:
        i+=1
        if cdf_normalized[i] > 0.01:
            flag = False
    winleft = int(0.8*(imgmin + int((i/Nbins)*(imgmax-imgmin))))
    flag = True
    i=0
    while flag:
        i+=1
        if cdf_normalized[i] > 0.99:
            flag = False
    winright = imgmin + int((i/Nbins)*(imgmax-imgmin))

    level = int((winleft + winright)/2)
    window = winright - winleft
    return window,level










# test function
if __name__ == "__main__":

    # 统计图像差别
    # import matplotlib.pyplot as plt
    # dcm = pydicom.read_file('/home/chenhao/device/method_test_save/WL/bone_result/307-1001168771.dcm')
    # w = dcm.__getattr__('WindowWidth')
    # l = dcm.__getattr__('WindowCenter')
    # img = dcm.pixel_array
    img = cv2.imread('/home/chenhao/device/method_test_save/WL/bone/CHNCXR_0001_0.png',cv2.IMREAD_UNCHANGED)
    window,level= get_W_L_simple_distribution(img)
    # w=dcm.__getattr__('WindowWidth')
    # l=dcm.__getattr__('WindowCenter')
    # print(w,l)
    print(window,level)
    # img = dcm.pixel_array
    # cdf = getCH(img)
    # plt.plot(range(256),cdf)
    # plt.show()
    # w,l = get_W_L_simple_distribution(img)
    # print(w,l)
    #img1 = cv2.imread('/home/chenhao/device/method_test_save/WL/bone/CHNCXR_0017_0.png',cv2.IMREAD_UNCHANGED)
    # img2 = cv2.imread('/home/chenhao/device/method_test_save/WL/bone/CHNCXR_0005_0.png',cv2.IMREAD_UNCHANGED)
    # cdf1 = getCH(img1)
    # cdf2 = getCH(img2)
    # fig = plt.figure(num=1)
    # plt.plot(range(256),cdf1,color='b',label='need')
    # plt.plot(range(256),cdf2,color='r',label='unneed')
    # plt.legend(loc = 'upper left')
    # plt.show()
    # # plt.hist(img1.flatten(),256,[img1.min(),img1.max()],color='r')
    # # plt.show()
    # window1,level1 = get_W_L_simple_distribution(img1)
    # window2,level2 = get_W_L_simple_distribution(img2)
    #
    # print(window1,level1)
    # print(window2,level2)

    # 整个函数效果测试 png图像
    # path = '/home/chenhao/device/method_test_save/WL/bone/'
    # lists = os.listdir(path)
    # for file in lists:
    #     img = cv2.imread(path + file,cv2.IMREAD_UNCHANGED)
    #     window,level = get_W_L_simple_distribution(img)
    #    # window,level = cal_window_level(img,cdf_config)
    #     print('{}`s window is {},level is {}'.format(file,window,level))

    #批测试dcm图像处理效果
    # path = '/home/chenhao/device/Data/去骨提高诊断率展示/bone/'
    # save_path = '/home/chenhao/device/Data/去骨提高诊断率展示/bone_result/'
    # lists = os.listdir(path)
    # for file in lists:
    #     dcm = pydicom.read_file(path+file)
    #     img = dcm.pixel_array
    #     window,level = get_W_L_simple_distribution(img)
    #     dcm.__setattr__('WindowCenter',level)
    #     dcm.__setattr__('WindowWidth',window)
    #     dcm.save_as(save_path+file)
    # dcm = pydicom.read_file('/home/chenhao/device/method_test_save/WL/Bone_dcm/X15207198.dcm')
    # dcm1 = pydicom.read_file('/home/chenhao/device/method_test_save/WL/Bone_dcm/X15207199.dcm')
    # img1 = dcm1.pixel_array
    # img = dcm.pixel_array
    # cdf = getCH(img)
    # cdf1 = getCH(img1)
    # plt.plot(range(256), cdf, color='b',label='wrong')
    # plt.plot(range(256),cdf1, color='r',label='right')
    # plt.legend(loc='upper left')
    # plt.show()
    # window,level = get_W_L_simple_distribution(img)
    # print(window,level)


