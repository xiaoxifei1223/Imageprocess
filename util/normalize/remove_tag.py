# coding: utf-8
# TODO: 用于去标签
# Author: chenhao
# data: 2018.11.26

# coding:utf-8

import numpy as np
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