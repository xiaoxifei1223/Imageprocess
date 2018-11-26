# coding:utf-8
'''
TODO: 进行图像修复
author: chenhao
data: 2018.10.31
email: haochen9212@outlook.com
'''


import cv2
import pydicom
import numpy as np
import numpngw

dcm = pydicom.read_file('/media/chenhao/B4FE-5315/display_test/rib_rm_lvl_1.dcm')
img = dcm.pixel_array
img = img.astype(np.uint16)
#img = cv2.resize(img, (1024, 1024))
# img = cv2.imread('/home/chenhao/device/Code/image-inpainting-master/test_images/test1/test.png')
# img = img[:,:,0]
#img = img.astype(np.uint16)
#img = cv2.resize(img, (1024,1024), cv2.INTER_LANCZOS4)
#img_new = 255.0 * img / (img.max()-img.min())
#img_new = img_new.astype(np.uint8)

mask = np.zeros((2688,2208))
mask[962:1016, 893:952] = 1
mask[9:168, 684:837] = 1
mask[2331:2430, 1980:2037] = 1
mask[114:237, 144:423] = 1
mask[2553:2685, 1923:2016]
mask = mask.astype(np.uint8)
# mask= cv2.imread('/home/chenhao/device/method_test_save/Restore/Result of test_DX_1.png')
# mask = mask[:,:,0]

img_re = cv2.inpaint(img,mask,4,cv2.INPAINT_TELEA)

numpngw.write_png('/home/chenhao/device/method_test_save/Restore/test.png',img_re)


