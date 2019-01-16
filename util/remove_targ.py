# coding:utf-8
'''
TODO: 用来去除标签
'''
import numpy as np
import pydicom
import numpngw
import cv2
import os

def _mask_expand(mask, kernel_type=5, iterations=2):
    '''
    TODO: 对mask进扩展，使用图像膨胀运算扩展mask中肺腔的大小
    :param mask: ndarray 保证输入mask是uint8的二值图像
    :return: mask_new ndarray 输出扩展后的mask区域
    '''
    kernel= np.ones((kernel_type, kernel_type), np.uint8)
    mask_new = cv2.dilate(mask, kernel, iterations=iterations)
    return mask_new

def getmean(img, mask):
    '''
    TODO: 根据mask标定的范围计算均值
    :param img:
    :param mask:
    :return:
    '''
    mask[mask != 0] = 255
    img_new = cv2.add(img, np.zeros(np.shape(img), dtype=img.dtype), mask=mask)
    img_new = img_new[img_new !=0]
    return np.mean(img_new)


def need_inverse(img, mask):
    '''
    TODO: 人工判断是否需要进行反向
    :param img: ndarray 输入图像
    :param mask: ndarray mask图像用于标注肺部区域
    :return: bool 给出是否需要进行反向
    '''
    # flag = False    # 默认不需要进行反向
    # row, col = img.shape
    # if row != col : # cv2的resize如果col和row不一样会将col放前面造成错误
    #     mask = cv2.resize(mask, (col, row))
    # else:
    #     mask = cv2.resize(mask, (row, col))
    # img = img.astype(np.uint16)
    # mask = mask.astype(np.uint8)
    # mask_expand = _mask_expand(mask)
    #
    # # 根据肺腔mask中的均值和扩张mask后的均值进行计算,如果扩张后的均值比扩张前的还小，说明周边一圈骨组织比肺内还小说明需要反向
    # mean_lung = getmean(img, mask)
    # mean_lung_expand = getmean(img, mask_expand)
    #
    # if mean_lung_expand < mean_lung:
    #     flag = True
    #     return flag
    # else:
    #     return flag
    row, col = img.shape
    img_block1 = img[0:100, 0:col]
    img_block2 = img[100:row-100, 0:col-100]
    img_block3 = img[row-100:row, 0:col]
    img_block4 = img[100:row-100, col-100:col]

    m1 = np.mean(img_block1)
    m2 = np.mean(img_block2)
    m3 = np.mean(img_block3)
    m4 = np.mean(img_block4)

    mean_total = np.mean(img)
    mean_judge = np.mean([m1, m2, m3, m4])
    Flag =False
    if mean_judge > mean_total:
        Flag = True
    else:
        Flag = False
    return Flag




def regin_repair(img, pos):
    '''
    TODO:将标签位置补上
    :param img: 原图
    :param pos: 位置
    :return:
    '''
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    for i, j in zip(pos[0], pos[1]):
        mask[i, j] = 1
    img = img.astype(np.uint16)
    img_new = cv2.inpaint(img, mask, 7, cv2.INPAINT_TELEA)

    return img_new

# 去标签
def removal_tag(ds):

    # 如果存在线形转换标签，则进行线形转换
    img = ds.pixel_array
    if hasattr(ds, "RescaleIntercept") and hasattr(ds, "RescaleSlope"):
        img = ds.RescaleSlope * img + ds.RescaleIntercept

    img = np.asarray(img, np.float)


    # 根据PresentationLUTShape 判断是否需要取反
    if hasattr(ds, "PresentationLUTShape") and hasattr(ds, "PhotometricInterpretation"):
        if ds[0x2050, 0x0020].value == "INVERSE" and ds[0x0028, 0x0004].value == "MONOCHROME1":

            vmin = img[img > 0].min()
            vmax = img.max()

            # 白色标签阈值
            whitetag = 0
            pos = np.where(img == whitetag)
            img = regin_repair(img, pos)
        else:
            # 白色标签阈值
            whitetag = pow(2, ds[0x0028, 0x0101].value) - 1
            whitetagwide = 5

            pos = np.where(img > (whitetag - whitetagwide))
            img = regin_repair(img, pos)


    else:
        # 白色标签阈值
        whitetag = pow(2, ds[0x0028, 0x0101].value) - 1
        whitetagwide = 5

        pos = np.where(img > (whitetag - whitetagwide))
        img = regin_repair(img, pos)

    return img


# 测试去标签算法
if __name__ == "__main__":
    # path = r"G:\Data\DicomImages\train2048\soft\X15207198.dcm"
    # save_path = '/home/chenhao/device/Data/DicomImages/train512/trainA/'
    # lists = os.listdir(path)
    # for file in lists:
    #     print('{} is processing'.format(file))
    #     dcm_path = os.path.join(path, file)
    #     dcm = pydicom.read_file(dcm_path)
    #     img = removal_tag(dcm)
    #     img_new = cv2.resize(img, (512, 512), cv2.INTER_LANCZOS4)
    #     img_new = img_new.astype(np.uint16)
    #     save_name = file.split('.')[0] + '.png'
    #     save_img_path = os.path.join(save_path, save_name)
    #     numpngw.write_png(save_img_path, img_new)
    # print('finished')
    dcm = pydicom.read_file(r"G:\Data\DicomImages\train2048\soft\X15207198.dcm")
    img = removal_tag(dcm)
    cv2.imwrite(r"G:\X15207198_soft.png", img)