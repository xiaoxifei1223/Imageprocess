# coding: utf-8

# TODO: 用于进行数据的产生
# Author: chenhao
# Data:2048.11.26

import numpy as np
import random
import cv2
import pydicom
from util.normalize.remove_tag import removal_tag
import os
import numpngw



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


def inverse_bone(bone, mask):

    vmax = bone.max()
    vmin = bone.min()

    bone = vmax + vmin - bone

    pos = np.where(mask == 0)

    for i, ii in zip(pos[0], pos[1]):
        bone[i, ii] = 100

    return bone


# 软组织图灰度增广
def generator(bone, soft, mask):
    ws = random.uniform(0.02, 0.98)
    wb = 1 - ws
    img_min = soft.min()
    img_max = soft.max()

    fsoft = nonlinearImg(soft, mask)
    bone = normalize(bone)
    fsoft = normalize(fsoft)
    newsrc = bone * wb + fsoft * ws
    # bone_test = (bone*wb)*(img_max - img_min)
    # bone_test = bone_test.astype(np.uint16)
    # numpngw.write_png('/home/chenhao/device/method_test_save/Normalize/test/generate_bone/' + str(i) + '.png',
    #                   bone_test)
    # fsoft_test = (fsoft*ws)*(img_max - img_min)
    # fsoft_test = fsoft_test.astype(np.uint16)
    # numpngw.write_png('/home/chenhao/device/method_test_save/Normalize/test/generate_soft/' + str(i) + '.png',
    #                   fsoft_test)
    newsrc = newsrc * (img_max - img_min)

    return newsrc, ws, wb




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
    else:
        s = np.random.randint(3,9)
        if s % 2 == 0:
            s = s+1
        img = cv2.GaussianBlur(img, (s,s), sigmaX=2, sigmaY=2)

    return img


# 软组织图训练灰度增广
def grayaug(bone, target, mask, threshold):

    # 如果使用的是双能源骨图， 需要取反， 使用生产的骨图则不需要取反
    # bone = inverse_bone(bone)      # 已经先进行处理

    # 使用软组织图与骨图合成图 现在target包含两部分[0]是软组织,[1]是骨组织
    newimg = generator(bone, target, mask)


    #  归一化合成图
    newimg = lungwin_mask(newimg, mask,  threshold)

    # 对合成图进行分辨率增广
    newimg = resolution(newimg)

    #  归一化软组织图
    soft = lungwin_mask(target, mask, threshold)


    return newimg, soft





def load_img(file, shape, path):
    targetpath = os.path.join(path, file)
    if targetpath.endswith("dcm"):
        target_dcm = pydicom.read_file(targetpath)
        target = removal_tag(target_dcm)
        target = cv2.resize(target, (shape[1], shape[0]), cv2.INTER_LANCZOS4)
        target = np.array(target, np.float)
    elif targetpath.endswith("png"):
        target = cv2.imread(targetpath, cv2.IMREAD_UNCHANGED)
        if len(target.shape) > 2:
            target = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
        target = cv2.resize(target, (shape[1], shape[0]), cv2.INTER_LANCZOS4)
        target = np.array(target, np.float)
    else:
        raise Exception("Data format is not dcm or png")

    return target


def normalize(img):
    '''
    TODO: 进行min-max归一化
    :param img:
    :return:
    '''
    img_min = img.min()
    img_max = img.max()
    img_new = (img - img_min) / (img_max - img_min)
    return img_new

if __name__ =="__main__":
    path_soft = '/home/chenhao/device/method_test_save/Normalize/test/soft/'
    path_bone = '/home/chenhao/device/method_test_save/Normalize/test/bone/'
    path_mask = '/home/chenhao/device/method_test_save/Normalize/test/mask/'

    test_path = '/home/chenhao/device/method_test_save/Normalize/test/generate_2/'

    shape = [512, 512]
    num = 100

    soft = load_img('X15228219.dcm', shape, path_soft)
    bone = cv2.imread('/home/chenhao/device/method_test_save/Normalize/test/bone/inverse_mask.png',
                      cv2.IMREAD_UNCHANGED)
    #bone = load_img('X15228219.dcm', shape, path_bone)
    # mask_inverse = cv2.imread('/home/chenhao/device/method_test_save/Normalize/test/output_mask/X15228219.tif',
    #                           cv2.IMREAD_UNCHANGED)
    # bone = inverse_bone(bone, mask_inverse)
    # bone = bone.astype(np.uint16)
    # numpngw.write_png('/home/chenhao/device/method_test_save/Normalize/test/bone/inverse_mask.png',
    #                   bone)
    mask = load_img('X15228219.png', shape, path_mask)
    #numpngw.write_png('/home/chenhao/device/method_test_save/Normalize/test/generate/bone.png', bone)
    #soft = soft.astype(np.uint16)
    #numpngw.write_png('/home/chenhao/device/method_test_save/Normalize/test/generate/soft.png', soft)

    for i in range(num):
        img_new ,ws, wb= generator(bone, soft, mask)
        print('num({}) {}, {}'.format(i, ws, wb))
        img_new = img_new.astype(np.uint16)
        save_path = os.path.join(test_path, str(i) + '.png')
        numpngw.write_png(save_path, img_new)
    print('finished')


























