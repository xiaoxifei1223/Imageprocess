# coding:utf-8
'''
TODO:用来根据肺部mask截断锁骨，获取锁骨肺腔内区域
'''

def cutClavicle(mask,mask_clavicle):
    temp = mask_clavicle
    temp[mask == 0] = 0
    return temp