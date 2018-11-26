# coding:utf-8
'''
TODO:用来从文件夹中提取数据
'''
import os
import shutil
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

if __name__ =="__main__":
    path = '/home/chenhao/device/method_test_save/forlable/'
    folders = os.listdir(path)
    save_path = '/home/chenhao/device/method_test_save/DR_NORMALIZE/'
    for folder in folders:
        path_dcm = os.path.join(os.path.join(path, folder), folder + '.dcm')
        if os.path.exists(path_dcm):
            print('{} is processing'.format(folder))
            save_path_dcm = os.path.join(save_path, folder + '.dcm')
            shutil.copy(path_dcm, save_path_dcm)
    print('ok')