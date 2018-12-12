# coding:utf-8
# test normalize method
# chenhao
import cv2
import pydicom
import numpy as np
from util.remove_targ import removal_tag
from albumentations.pytorch.functional import img_to_tensor #使用最新的albumentation版本

def standardize(img_soft, img_withbone, threshold, mask=None, vthreshold=None):


    if mask is not None:
        # lung_org = img_blend(image_org, img_withbone, mask)
        lung_soft = img_soft * mask
        lung_withbone = img_withbone * mask
    else:
        lung_soft = img_soft
        lung_withbone = img_withbone

    if vthreshold is None:
        lung_soft_min = lung_soft[lung_soft > 0].min()
        lung_soft_max = lung_soft.max()

        lung_withbone_min = lung_withbone[lung_withbone > 0].min()
        lung_withbone_max = lung_withbone.max()

        #vthreshold = (lung_org_min, lung_org_max)
    else:
        lung_soft_min = vthreshold[0]
        lung_soft_max = vthreshold[1]
        lung_withbone_min = vthreshold[0]
        lung_withbone_max = vthreshold[1]
    # 对target进行归范化处理
    a_org = (threshold[1] - threshold[0]) / (lung_soft_max - lung_soft_min)
    b_org = (threshold[0] * lung_soft_max - lung_soft_min * threshold[1]) / (lung_soft_max - lung_soft_min)

    image_soft = img_soft * a_org + b_org
    image_soft[image_soft < threshold[0]] = threshold[0]
    image_soft[image_soft > threshold[1]] = threshold[1]

    # 对人造原片进行归范化处理
    withbone_a_org = (threshold[1] - threshold[0]) / (lung_withbone_max - lung_withbone_min)
    withbone_b_org = (threshold[0] * lung_soft_max - lung_withbone_min * threshold[1]) / (lung_withbone_max - lung_withbone_min)

    image_withbone = img_withbone * withbone_a_org + withbone_b_org
    image_withbone[image_soft < threshold[0]] = threshold[0]
    image_withbone[image_soft > threshold[1]] = threshold[1]



    return img_to_tensor(image_withbone), img_to_tensor(image_soft)



if __name__ =="__main__":
    dcm_soft = pydicom.read_file(r"G:\Data\DicomImages\train2048\soft\X15207198.dcm")
    dcm_bone = pydicom.read_file(r"G:\Data\DicomImages\train2048\src\data\X15207198.dcm")

    img_soft = removal_tag(dcm_soft)
    img_bone = removal_tag(dcm_bone)

    mask = cv2.imread(r"G:\Data\DicomImages\train2048\mask\data\X15207198.png", cv2.IMREAD_UNCHANGED)
    mask = cv2.resize(mask, (img_soft.shape[0], img_soft.shape[1]))
    img_bone,img_soft = standardize(img_soft, img_bone, [img_soft.min(),img_soft.max()], mask)

    img_bone = img_bone.astype(np.uint16)
    img_soft = img_soft.astype(np.uint16)
    cv2.imwrite('G:/test_bone.png', img_bone)
    cv2.imwrite('G:/test_soft.png', img_soft)
