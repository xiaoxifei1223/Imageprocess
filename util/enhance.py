# coding: utf-8
'''
TODO: 图像增强
'''
import cv2
import numpngw
import pydicom
import numpy as np
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
    dcm = pydicom.read_file(r"G:\method_test_save\Rib_Aug\X15209072.dcm")
    img = dcm.pixel_array
    img = img.astype(np.uint16)
    img_new = enhance_core(img, 2.0, (4, 4))
    numpngw.write_png(r'G:\method_test_save\Rib_Aug\result\test_bone.png', img_new)