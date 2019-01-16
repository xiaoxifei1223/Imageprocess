# coding:utf-8
# TODO: 用来删除文件

import os

path_ref = 'G:\Data\Rib_2_soft_data\data'
path_del = 'G:\Data\Rib_2_edge_mask\data'

lists_ref = os.listdir(path_ref)
lists_del = os.listdir(path_del)

is_same_end = False

lists_ref_name = list(map(lambda s: s.split('.')[0], lists_ref))
lists_del_name = list(map(lambda s: s.split('.')[0], lists_del))

lists_ref_end = list(map(lambda s: s.split('.')[-1], lists_ref))
lists_del_end = list(map(lambda s: s.split('.')[-1], lists_del))

for file in lists_del_name:
    if file not in lists_ref_name:
        index = lists_del_name.index(file)
        file_name = file + '.' + lists_del_end[index]
        path = os.path.join(path_del, file_name)
        os.remove(path)

