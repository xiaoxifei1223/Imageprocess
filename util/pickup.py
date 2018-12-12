# coding:utf-8
'''
TODO:用来从文件夹中提取数据
'''
import os
import shutil
import pydicom
import cv2
import numpngw
import numpy as np
from util.normalize.remove_tag import removal_tag
#
# if __name__ =="__main__":
#     path1 = '/media/chenhao/Elements/chenhao/TB_DATA/TB_png/data/'
#     path2 = '/media/chenhao/Elements/chenhao/TB-ImgBoxes1/'
#     ref_path = '/media/chenhao/Elements/chenhao/TB-ImgBoxes1/'
#     save_path = '/home/chenhao/device/Data/病例图像/'
#
#     reffiles = os.listdir(ref_path)
#     for file in reffiles:
#         if os.path.exists(path1+file):
#             shutil.copy(path1+file,save_path+file)
#         elif os.path.exists(path2+file):
#             shutil.copy(path2+file,save_path+file)
#         else:
#             print('{} no found'.format(file))
#     print('finished')

# 直接从文件夹里面提取数据
# if __name__ =="__main__":
#     path = '/home/chenhao/device/method_test_save/forlable/'
#     folders = os.listdir(path)
#     save_path = '/home/chenhao/device/method_test_save/DR_NORMALIZE/'
#     for folder in folders:
#         path_dcm = os.path.join(os.path.join(path, folder), folder + '.dcm')
#         if os.path.exists(path_dcm):
#             print('{} is processing'.format(folder))
#             save_path_dcm = os.path.join(save_path, folder + '.dcm')
#             shutil.copy(path_dcm, save_path_dcm)
#     print('ok')
def getfilesfromtxt(txtpath):

    if os.path.exists(txtpath):
        filelist = []
        with open(txtpath,"rt") as fp:
            files = fp.readlines()
            for file in files:
                fileitem = file.split("\n")[0]
                if fileitem.endswith("dcm") or fileitem.endswith("png"):
                    filelist.append(fileitem)
                else:
                    raise Exception("no png or dcm files in directory")

        return filelist
    else:
        raise Exception("The file {} is not exist".format(txtpath))

# 通过读入的file.list 来提取数据
if __name__ =="__main__":
    file_path = r"G:\Data\file_bone.txt"
    path = r'G:\Data\DicomImages\train2048\bone_inverse'
    save_root_path = r'G:\Data\DicomImages\train2048\bone_inverse_out'
    filelists = getfilesfromtxt(file_path)
    for file in filelists:
        dcm_path = os.path.join(path, file)
        dcm = pydicom.read_file(dcm_path)
        #img = removal_tag(dcm)
        #img = cv2.resize(img, (512, 512))
        #img = img.astype(np.uint16)
        save_img_path = os.path.join(save_root_path, file.split('.')[0] + '.dcm')
        #dcm.PixelData = img.tobytes()
        dcm.save_as(save_img_path)
    print('finished')


