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
#from util.normalize.remove_tag import removal_tag
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
   txt_path = r"/home/chenhao/home/code/ImageProcess/file.txt"
   lists = getfilesfromtxt(txt_path)
   save_path = r'/home/chenhao/home/code/ImageProcess/pickup/'
   path1 = '/data/TB/TB_lyz/train/image/src/'
   # #path2 = '/data/TB/tb-voc/dicomImages/'
   # #path3 = '/data/TB/TB_lyz/train/image/src/'
   count= 0
   isMask = False
   for file in lists:
       img_path = os.path.join(path1, file)
       if isMask:
           if img_path.endswith('.dcm'):
               img_path = img_path.split('.dcm')[0] + '.png'
           if os.path.exists(img_path):
               print(file)
               count +=1
               img_save_path = os.path.join(save_path, file.split('.dcm')[0] + '.png')
               shutil.copy(img_path, img_save_path)
       else:
           if os.path.exists(img_path):
               print(file)
               count +=1
               img_save_path = os.path.join(save_path, file)
               shutil.copy(img_path, img_save_path)

   print(count)
   # path1 = '/data/TB/for _new_mask/'
   # lists = os.listdir(path1)
   # for file in lists:
   #     img_path = os.path.join(path1,file)
   #     print(img_path)
   #     if img_path.endswith('.dcm'):
   #         # dcm = pydicom.read_file(img_path)
   #         # img = dcm.pixel_array
   #         # img_new_path = img_path.split('.dcm')[0] + '.png'
   #         # cv2.imwrite(img_new_path,img)
   #         os.remove(img_path)


