# coding:utf-8
# TODO: 用来写记录图像的txt文本
import os


def record_name(path,savepath):
    lists = os.listdir(path)
    f = open(savepath,'w')
    for file in lists:
        f.write(str(file)+'\n')
    f.close()


if __name__ =="__main__":
    path = r"G:\Data\DicomImages\Rib_2_soft_data\data"
    save_path = r"G:\Data\DicomImages\Rib_2_soft_data\soft_name_list.txt"
    record_name(path, save_path)
