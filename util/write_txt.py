# coding:utf-8
import os

def write_txt(path, savepath):
    lists = os.listdir(path)
    f = open(savepath, 'w')
    for file in lists:
        f.write(str(file)+'\n')
    f.close()


if __name__ =="__main__":
    path = r'G:\method_test_save\Image disturb\soft'
    savepath = r'G:\method_test_save\Image disturb\soft\file.txt'
    write_txt(path, savepath)