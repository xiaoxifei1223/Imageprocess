# coding --utf8
'''
author:chenhao
data:2018.07.20
TODO：将png数据转为1024大小的16位png数据
'''

import cv2
import os
import png
import pydicom
import pdb
import numpy as np


#path = 'F:/Data/src_affine/data/'
#savepath = 'F:/Data/src_affine_1024'

#imgShape= (1024,1024)

#listdir = os.listdir(path)


def save_png16bit_grayscale(y, filename):
    # Convert y to 16 bit unsigned integers.
    #z = (65535 * ((y - y.min()) / y.ptp())).astype(np.uint16)
    #
    #zgray = z
    z = y
    zgray = z
    # Here's a grayscale example.
    # Use pypng to write zgray as a grayscale PNG.
    with open(filename, 'wb') as f:
        writer = png.Writer(width=y.shape[1], height=y.shape[0], bitdepth=16, greyscale=True)
        zgray2list = zgray.tolist()
        writer.write(f, zgray2list)

# for file in listdir:
#     imgpath = os.path.join(path,file)
#     img = cv2.imread(imgpath,cv2.IMREAD_ANYDEPTH)
#     imgnew = cv2.resize(img,imgShape,interpolation=cv2.INTER_LANCZOS4)
#     saveimg = os.path.join(savepath,file)
#     save_png16bit_grayscale(imgnew,saveimg)
#     print('{} is processing'.format(file))

def img_resize(img,img_shape):
    imgnew = cv2.resize(img,img_shape,interpolation=cv2.INTER_LANCZOS4)
    return imgnew

if __name__ == '__main__':
    path = '/home/chenhao/device/Data/DicomImages/train512/src/'
    savepath = '/home/chenhao/device/Data/DicomImages/train512/src_png/'

    imgShape = (512, 512)
    # img = cv2.imread(path,cv2.IMREAD_ANYDEPTH)
    # rc = RemapIntensity()
    # img = rc.run(img)
    # save_png16bit_grayscale(img, savepath)

    listdir = os.listdir(path)
    for file in listdir:
        imgpath = os.path.join(path,file)
        img = pydicom.read_file(imgpath).pixel_array
        imgnew = cv2.resize(img,imgShape,interpolation=cv2.INTER_LANCZOS4)
        saveimg = os.path.join(savepath,file.split('.dcm')[0] + '.png')
        save_png16bit_grayscale(imgnew,saveimg)
        print('{} is processing'.format(file))






