# coding:utf8
'''
TODO: 图像反相
'''
import numpy as np
import os
import pydicom

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
            h, w = img.shape
            if len(pos[0]) > 0:
                x, y = pos[0][0], pos[1][0]
                x0 = max(0, x-10)
                y0 = max(0, y-10)
                x1 = min(w, x+10)
                y1 = min(h, y+10)
                area = img[x0:x1,y0:y1]
                if len(area[area > whitetag]) > 0:
                    value = area[area > whitetag].mean()
                else:
                    value = img[img > whitetag].mean()

                img[img == whitetag] = value

            img = vmax + vmin - img

        else:
            # 白色标签阈值
            whitetag = pow(2, ds[0x0028, 0x0101].value) - 1
            whitetagwide = 5

            pos = np.where(img > (whitetag - whitetagwide))
            h, w = img.shape

            if len(pos[0]) > 0:
                x, y = pos[0][0], pos[1][0]
                x0 = max(0, x - 10)
                y0 = max(0, y - 10)
                x1 = min(w, x + 10)
                y1 = min(h, y + 10)
                area = img[x0:x1, y0:y1]
                if len(area[area < (whitetag - whitetagwide)]) > 0:
                    value = area[area < (whitetag - whitetagwide)].mean()
                else:
                    value = img[img < (whitetag - whitetagwide)].mean()

                img[img > (whitetag - whitetagwide)] = value

    else:
        # 白色标签阈值
        whitetag = pow(2, ds[0x0028, 0x0101].value) - 1
        whitetagwide = 5

        pos = np.where(img > (whitetag - whitetagwide))
        h, w = img.shape

        if len(pos[0]) > 0:
            x, y = pos[0][0], pos[1][0]
            x0 = max(0, x - 10)
            y0 = max(0, y - 10)
            x1 = min(w, x + 10)
            y1 = min(h, y + 10)
            area = img[x0:x1, y0:y1]
            if len(area[area < (whitetag - whitetagwide)]) > 0:
                value = area[area < (whitetag - whitetagwide)].mean()
            else:
                value = img[img < (whitetag - whitetagwide)].mean()

            img[img > (whitetag - whitetagwide)] = value

    return img

def inverse(img):
    img_max = img.max()
    r,c = img.shape[0],img.shape[1]
    img_re = np.zeros((r,c))
    for i in range(r):
        for ii in range(c):
            img_re[i,ii] = img_max - img[i,ii]
    factor = abs(img_re.mean() - img.mean())
    img_re = img_re - factor
    img_re[img_re < 0.0] = 0
    return img_re

def _png2dicom(imgpng,dcm):
    dcm.PixelData = imgpng
    return dcm

def img_dcm(img,dcm,savepath):
    if dcm.__getattr__('BitsAllocated') == 16:
        img = img.astype(np.uint16)
    elif dcm.__getattr__('BitsAllocated') == 8:
        img = img.astype(np.uint8)
    else:
        raise Exception("bits allocated is {}".format(dcm.__getattr__('BitsAllocated')))
    dcm = _png2dicom(img,dcm)
    dcm.save_as(savepath)


if __name__ =="__main__":
    root = '/home/chenhao/187_data/RibRemove/dual-cxr/DicomImages/train2048/bone/data'
    files = os.listdir(root)
    savepath = '/home/chenhao/device/Data/DicomImages/train2048/bone_inverse/'
    for file in files:
        print('{} is processing'.format(file))
        dcm = pydicom.read_file(os.path.join(root,file))
        img = removal_tag(dcm)
        img = inverse(img)
        img_dcm(img,dcm,os.path.join(savepath,file))
        print('{} is processed'.format(file))