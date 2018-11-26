# coding:utf-8
import cv2
import numpngw


def gaussian_blur(img,k_size):
    img_re = cv2.GaussianBlur(img,k_size,sigmaX=5,sigmaY=5)
    return img_re

if __name__ =="__main__":
    img = cv2.imread('/home/chenhao/device/Data/DicomImages/train2048/src_2048/data/X15207198.png',cv2.IMREAD_UNCHANGED)
    im_new = gaussian_blur(img,(13,13))
    numpngw.write_png('/home/chenhao/test.png',im_new)
