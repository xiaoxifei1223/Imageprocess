# coding:utf8
'''
TODO: 主函数
'''
import cv2
from util.imginverse import inverse
from util.remove_targ import removal_tag
from util.record import readfile
from pathos.multiprocessing import ProcessingPool as Pool
import os
import pydicom
import numpngw
import numpy as np
from util.cutClavicle import cutClavicle
from util.pyramid import grayAdPy,getfile
from util.common import save_png
from util.png_dicom import png2dicom

if __name__ =="__main__":
    # ## 进行多尺度灰度调整
    # root = '/home/chenhao'
    # path = os.path.join(root,'device/method_test_save/grayAd/')
    # savepath = os.path.join(root,'device/method_test_save/grayAd/result/')
    #
    # list1,list2,length = getfile(path)
    #
    # for i in range(length):
    #     print('{} is processing'.format(list1[i]))
    #     img = grayAdPy(list1[i],list2[i],max_level=8,filter_size=5)
    #     savename = list1[i].split('/')[-1]
    #     savename = savepath + savename.split('.')[0] + '.png'
    #     save_png(savename,img)
    #     print(list1[i])
    # files = os.listdir('/home/chenhao/device/method_test_save/grayAd/withbone')
    # for file in files:
    #     name = '/home/chenhao/device/method_test_save/grayAd/withbone/' + file
    #     dcm = pydicom.read_file(name)
    #     #dcm307 = pydicom.read_file('/home/chenhao/device/method_test_save/307-20180614137_withbone.dcm')
    #     #img = removal_tag(dcm307)
    #     img = removal_tag(dcm)
    #     img = img.astype(np.uint16)
    #     # img = img.astype(np.uint16)
    #     # img = dcm307.pixel_array
    #     numpngw.write_png('/home/chenhao/device/method_test_save/grayAd/withbone_png/'+ file.split('.')[0] + '.png',img)
    #     print(file)

    ## 将png转为dcm
    path = '/home/chenhao/device/method_test_save/grayAd/'
    savepath = '/home/chenhao/device/method_test_save/grayAd/result/'
    png2dicom(path,savepathdir=savepath)





