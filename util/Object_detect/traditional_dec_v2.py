# coding:utf-8
# TODO: 使用局域最大值点来进行分割
import cv2
from skimage.feature import peak_local_max
import os
import numpy as np

def findlocalmax(img,radius):
    cordinate = peak_local_max(img,min_distance=radius)
    return cordinate

if __name__ == "__main__":
    # path = 'F:\TB\HeatImg'
    # savepath = 'G:\method_test_save\Object_test\src'
    # lists = os.listdir(path)
    # for file in lists:
    #     img = cv2.imread(os.path.join(path,file))
    #     img = img[:,:,0]*0.299 + img[:,:,1]*0.587 + img[:,:,2]*0.114  # RGB转灰度公式
    #     img[img > 255] = 255
    #     img = img.astype(np.uint8)
    #     cv2.imwrite(os.path.join(savepath,file),img)

    img = cv2.imread(r"G:\method_test_save\Object_test\src\23283_0.png",
                     cv2.IMREAD_UNCHANGED)

    cordinate = findlocalmax(img, 100)
    mask = []
    for i in range(cordinate.shape[0]):
        if img[cordinate[i,1],cordinate[i,0]] != 255:
           mask.append(i)

    for i in range(len(mask)):
        cordinate = np.delete(cordinate,mask[i],axis=0)

    import matplotlib.pyplot as plt
    plt.scatter(cordinate[:,1], cordinate[:,0])
    plt.show()

