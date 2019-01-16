# coding:utf-8
import cv2


def otsu_seg(img):
    ret, th = cv2.threshold(img,0,255,cv2.)