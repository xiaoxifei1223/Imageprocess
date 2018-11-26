# coding: utf-8

# TODO: 用于进行数据的产生
# Author: chenhao
# Data:2048.11.26

import numpy as np
import random
import cv2

def lungwin_mask(image_org, mask, threshold):

    if mask is not None:
        lung_org = image_org * mask
    else:
        lung_org = image_org

    lung_org_min = lung_org[lung_org>0].min()
    lung_org_max = lung_org.max()

    a_org = (threshold[1] - threshold[0]) / (lung_org_max - lung_org_min)
    b_org = (threshold[0] * lung_org_max - lung_org_min * threshold[1]) / (lung_org_max - lung_org_min)

    image_org = image_org * a_org + b_org

    image_org[image_org < threshold[0]] = threshold[0]
    image_org[image_org > threshold[1]] = threshold[1]

    return image_org


def nonlinearImg(soft, mask):
    if mask is not None:
        softtemp = soft * mask
    else:
        softtemp = soft

    vmax = softtemp[softtemp > 0].max()
    vmin = softtemp[softtemp > 0].min()

    soft_norm = soft.copy()
    soft_norm[soft_norm > vmax] = vmax
    soft_norm[soft_norm < vmin] = vmin
    soft_norm = (soft_norm - vmin) / (vmax - vmin)

    It = random.randint(5, 20)
    alpha_sum = 0
    v = 0.4
    if random.uniform(0, 1) > 0.5:
        n = random.uniform(v, 1.0 / v)
        alpha = random.uniform(0.01, 1.0)
        newsoft = alpha * np.power(soft_norm, n)
        alpha_sum += alpha
        for i in range(It - 1):
            n = random.uniform(v, 1.0 / v)
            alpha = random.uniform(0.01, 1.0)
            newsoft += alpha * np.power(soft_norm, n)
            alpha_sum += alpha

        newsoft = newsoft / alpha_sum

        newsoft = newsoft * (vmax - vmin) + vmin
        newsoft[soft == 0] = 0
    else:
        newsoft = soft


    return newsoft


def inverse_bone(bone):

    vmax = bone.max()
    vmin = bone.min()

    bone = vmax + vmin - bone

    return bone


# 软组织图灰度增广
def generator(bone, soft, mask):
    ws = random.uniform(0.02, 0.98)
    wb = 1 - ws

    fsoft = nonlinearImg(soft, mask)

    newsrc = bone * wb + fsoft * ws
    return newsrc




def resolution(img):
    '''
    TODO: 对图像进行分辨率增广
    :param img:
    :return:
    '''
    if np.random.uniform(0,1) > 0.5 :
        # if np.random.uniform(0.0,1.0) > 0.5:
        shape = img.shape
        img =cv2.resize(img, (shape[1]//2, shape[0]//2))
        img = cv2.resize(img, (shape[1], shape[0]))
        # else:
        #     s = np.random.randint(3,9)
        #     if s % 2 == 0:
        #         s = s+1
        #     img = cv2.GaussianBlur(img,(s,s),sigmaX=2,sigmaY=2)

    return img


# 软组织图训练灰度增广
def grayaug(bone, target, mask, threshold):

    # 如果使用的是双能源骨图， 需要取反， 使用生产的骨图则不需要取反
    #bone = inverse_bone(bone)      # 已经先进行处理

    # 使用软组织图与骨图合成图 现在target包含两部分[0]是软组织,[1]是骨组织
    newimg = generator(bone, target, mask)


    #  归一化合成图
    newimg = lungwin_mask(newimg, mask,  threshold)

    # 对合成图进行分辨率增广
    newimg = resolution(newimg)

    #  归一化软组织图
    soft = lungwin_mask(target, mask, threshold)

    return newimg, soft


# compressed模式下进行灰度增广
def grayaug_compressed(bone, soft, mask, threshold):

    # 使用软组织图与骨图进行合成
    new_img = generator(bone, soft, mask)   # new_img 是合成图像

    # 归一化合成图
    new_img = lungwin_mask(new_img, mask, threshold)

    # 进行分辨率增广
    new_img = resolution(new_img)

    # 归一化软组织图
    soft = lungwin_mask(soft, mask, threshold)

    # 归一化骨组织图
    bone = lungwin_mask(soft, mask, threshold)

    return new_img,soft,bone























