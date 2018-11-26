# coding: utf-8
'''
TODO: 图像增强
'''
import cv2
import numpngw
def enhance_core(input,clipLimit,tileGridSize):
    '''
    TODO: image local enhance and resize core
    :param input: tuple input[0] image array, input[1] center coordinates
    :return result: ndarray processed result
    '''

    img = input
    clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    img_re = clahe.apply(img)

    return img_re

if __name__ =="__main__":
    img = cv2.imread('/home/chenhao/device/method_test_save/Crop/soft/307-20180614119.png', cv2.IMREAD_UNCHANGED)
    img_new = enhance_core(img, 2.0, (4, 4))
    numpngw.write_png('/home/chenhao/device/method_test_save/Crop/test.png', img_new)