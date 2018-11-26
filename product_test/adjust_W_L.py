# coding:utf-8
'''
TODO:用来测试自适应的调整窗宽窗位算法测试代码
author: chenhao
data: 2018.10.24
email: haochen9212@outlook.com
'''
import numpy as np
import pydicom
import cv2
# 配置参数
W_L_config={'ad_factor':0.5,    # window衰减因子
            'Nbins':256,    # 求累积分布函数的bins数目
            'W_left_factor':0.01,   # 求window左边界阈值因子
            'W_right_factor':0.99,   # 求window右边界阈值因子
            'Area_factor':0.001     # 面积比例，避免过小值对累积函数的影响
            }


# 获得图像的累积分布函数
def getCH(img,nbins=256,th=0.001):
    '''
    TODO: get img Cumulative distribution histogram
    :param img: ndarray input image
    :param nbins: integer histogram bins
    :return ch： ndarray result of Cumulative distribution histogram
    '''
    # get image histogram
    imgmax = img.max()
    imgmin = img.min()

    hist,bins = np.histogram(img.flatten(),nbins,[imgmin,imgmax])
    # maybe cutdown histogram
    area = img.shape[0]*img.shape[1]
    if hist[0]/(img.shape[0]*img.shape[1]) > th:
        area = area-hist[0]
        hist[0] = 0

    # calculate cumulative histogram
    cdf = hist.cumsum()
    cdf_normalized = cdf /area   # get normalized cumulative distribution histogram

    return cdf_normalized,cdf,hist,bins


# 进行窗宽窗位获取的主函数
def get_W_L(img,config):
    '''
    TODO: get suitable gray image window and level
    :param img: ndarray input gray image
    :param config: dict parameters set
    :return : tuple
    '''
    imgmax = img.max()
    imgmin = img.min()

    Nbins = config['Nbins']
    th = config['Area_factor']

    cdf_normalized,cdf,hist,nbins = getCH(img,Nbins,th)
    flag = True
    i=0
    while flag:
        i+=1
        if cdf_normalized[i] > config['W_left_factor']:
            flag = False
    winleft = int(config['ad_factor']*(imgmin + (i/Nbins)*(imgmax-imgmin)))
    flag = True
    i=0
    while flag:
        i+=1
        if cdf_normalized[i] > config['W_right_factor']:
            flag = False
    winright = imgmin + int((i/Nbins)*(imgmax-imgmin))

    level = int((winleft + winright)/2)
    window = winright - winleft
    return window,level,winleft,winright,cdf_normalized,cdf,hist,nbins


# 测试主函数
def test(img,config):
    window, level, winleft, winright, cdf_normalized, cdf, hist, nbins = get_W_L(img,config)
    # 对计算出的立即数进行打印
    with open('record_wl.txt','w') as f:
        f.write('window is ')
        f.write(str(window) + '\t')
        f.write('level is ')
        f.write(str(level) + '\t')
        f.write('winleft is ')
        f.write(str(winleft)+ '\t')
        f.write('winright is ')
        f.write(str(winright)+ '\n')

        f.close()

    # 对计算出的序列进行打印
    np.savetxt('record_wl_cdfN.txt',cdf_normalized)
    np.savetxt('record_wl_cdf.txt',cdf)
    np.savetxt('record_wl_hist.txt',hist)
    np.savetxt('record_wl_nbins.txt',nbins)

# 运行测试函数

if __name__ =="__main__":
    imgname = '/home/chenhao/device/method_test_save/WL/bone_result/X15208876.png'
    if imgname.endswith('dcm'):
        dcm = pydicom.read_file(imgname)
        img = dcm.pixel_array
        test(img,W_L_config)
    elif imgname.endswith('.png'):
        img = cv2.imread(imgname,cv2.IMREAD_UNCHANGED)
        test(img,W_L_config)
    else:
        raise Exception('unsupported image type')



