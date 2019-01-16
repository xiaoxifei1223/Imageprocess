# coding:utf8
'''
TODO: 主函数
'''
import cv2

import os
import shutil
import pydicom
import numpngw
import numpy as np
# from util.cutClavicle import cutClavicle
# from util.pyramid import grayAdPy,getfile
# from util.common import save_png
# from util.png_dicom import png2dicom

if __name__ =="__main__":
  path = 'G:\Data\Rib_2_mask_data\data'
  save_path = r'G:\Data\Rib_2_mask_data\tem'
  lists = os.listdir(path)
  for file in lists:
      if file.endswith('.dcm'):
          file_name = os.path.splitext(file)
          file_name = file_name[0] + '.png'
          shutil.copy(os.path.join(path, file), os.path.join(save_path, file_name))
