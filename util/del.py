# coding:utf-8
'''
TODO: 用来进行某些图像的删除
'''

import os

def define_del(path1,path2):
    # 根据path1来判断，如果path1中有而path2中没有则删掉path1的内容
    if (os.path.exists(path1) == True) and (os.path.exists(path2) == False):
        return True
    else:
        return False


if __name__ == "__main__":
    path_del = '/home/chenhao/device/Data/TB_mark/data/'    # 待删掉的内容路径
    path_refer = '/home/chenhao/device/Data/TB_mark/png_2/'   # 参考内容路径

    lists = os.listdir(path_del)
    for file in lists:
        path1 = os.path.join(path_del, file)
        path2 = os.path.join(path_refer, file.split('.')[0] + '.png')
        need_del = define_del(path1, path2)
        if need_del:
            print(path1)
            os.remove(path1)