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



def lungwin_mask(image_org, mask, threshold, vthreshold=None):

    if mask is not None:
        #lung_org = img_blend(image_org, img_withbone, mask)
        lung_org = image_org * mask
    else:
        lung_org = image_org

    if vthreshold is None:
        lung_org_min = lung_org[lung_org>0].min()
        lung_org_max = lung_org.max()

        vthreshold = (lung_org_min, lung_org_max)
    else:
        lung_org_min = vthreshold[0]
        lung_org_max = vthreshold[1]

    a_org = (threshold[1] - threshold[0]) / (lung_org_max - lung_org_min)
    b_org = (threshold[0] * lung_org_max - lung_org_min * threshold[1]) / (lung_org_max - lung_org_min)

    image_org = image_org * a_org + b_org

    image_org[image_org < threshold[0]] = threshold[0]
    image_org[image_org > threshold[1]] = threshold[1]

    return image_org, vthreshold


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


    return newsoft, (vmin, vmax)


def inverse_bone(bone):

    vmax = bone.max()
    vmin = bone.min()

    bone = vmax + vmin - bone

    return bone

def crop_bone_body(bone, mask):
    '''
    TODO: 对bone的人体组织外围进行规范化处理，使得人体外的值不可能大于人体内
    :param bone:
    :param mask:
    :return: new_bone
    '''


    pos = np.where(mask == 0)
    #body_min = bone[pos].min()
    for i, ii in zip(pos[0], pos[1]):
        bone[i, ii] = 100

    return bone



def generator(bone, soft, mask, threshold, inverse_mask):
    '''
    TODO: 进行generator的测试，查看测试得到的合成图是否有问题
    :param bone:
    :param soft:
    :param mask:
    :param threshold:
    :param inverse_mask:
    :return:
    '''
    bone = crop_bone_body(bone, inverse_mask)

    ws = random.uniform(0.02, 0.98)
    wb = 1 - ws

    # bone = inverse_bone(bone)
    fsoft, vthreshold = nonlinearImg(soft, mask)
    cv2.imwrite('G:/test_soft.png', fsoft.astype(np.uint16))
    #fsoft = cv2.imwrite('G:/test_soft.png', (fsoft*65535).astype(np.uint16))
    newsrc = (bone * (wb) + fsoft * (ws))
    return newsrc




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


# 测试合成图像是否正常
if __name__ =="__main__":
    bone_path = r'G:\Data\DicomImages\train2048\bone_inverse_out'
    soft_path = r'G:\Data\DicomImages\train2048\soft_remove_tag'
    mask_inverse_path = r'G:\Data\DicomImages\train2048\mask_inverse\data'
    save_folder_path = r'G:\method_test_save\test_compose'
    mask = None
    num = 2
    lists = os.listdir(bone_path)
    for i in range(num):
        for file in lists:
            img_bone = pydicom.read_file(os.path.join(bone_path, file)).pixel_array.astype(np.uint16)
            img_soft = pydicom.read_file(os.path.join(soft_path, file)).pixel_array.astype(np.uint16)
            img_mask_inverse_path = os.path.join(mask_inverse_path, str(file).replace('.dcm', '.png'))
            img_mask_inverse = cv2.imread(img_mask_inverse_path, cv2.IMREAD_UNCHANGED)
            img_mask_inverse = cv2.resize(img_mask_inverse, (2048, 2048))
            img_bone = cv2.resize(img_bone, (2048, 2048))
            img_soft = cv2.resize(img_soft, (2048, 2048))



            new_img = new_img.astype(np.uint16)
            save_path = os.path.join(save_folder_path, file.split('.')[0]+'_'+str(num)+'.png')
            cv2.imwrite(save_path, new_img)

    print('finished')























 new_img = generator(img_bone, img_soft, mask, None,img_mask_inverse)