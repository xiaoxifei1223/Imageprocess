# coding:utf-8
'''
用来解析文件夹下的文件名
'''
import os

def readfile(path):
    lines = []
    with open(path,'r') as f:
        while True:
            file = f.readline()
            file = file.split('\n')[0]

            lines.append(file)
            if not file:
                break
                pass
    f.close()
    return lines



