# coding  --utf8
'''
author:chenhao
email:haochen9212@outlook.com
data:2017.7.30
'''
import numpy as np
import scipy
import scipy.signal
from skimage.color import rgb2gray
from scipy.misc import imread as imread
from scipy import linalg as linalg
import matplotlib.pyplot as plt
import os
import cv2
import png
import pydicom
from util.erode_dilation import enlarge_mask

def read_image(filename, representation=1):
    """
    A method for reading an image from a path and loading in as gray or in color
    :param filename: The path for the picture to be loaded
    :param representation: The type of color the image will be load in. 1 for gray,
    2 for color
    :return: The loaded image
    """

    # if representation == 1:
    #     # converting to gray
    #     im = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)
    #     im = im / 65535
    # else:
    #     if representation == 2:
    #         im = cv2.imread(filename, cv2.IMREAD_COLOR)
    #         # setting the image's matrix to be between 0 and 1
    #         im = im / 65535
    # return im

    im = imread(filename)
    if representation == 1:
        # converting to gray
        im = rgb2gray(im) / 255
    else:
        if representation == 2:
            im = im.astype(np.float64)
            # setting the image's matrix to be between 0 and 1
            im = im / 255
    return im


def create_gaussian_line(size):
    """
    A helper method for creating a gaussian kernel 'line' with the input size
    :param size: The size of the output dimension of the gaussian kernel
    :return: A discrete gaussian kernel
    """
    bin_arr = np.array([1, 1])
    org_arr = np.array([1, 1])
    if (size == 1):
        # special case, returning a [1] matrix
        return np.array([1])
    for i in range(size-2):
        # iterating to create the initial row of the kernel
        bin_arr = scipy.signal.convolve(bin_arr, org_arr)
    bin_arr = np.divide(bin_arr, bin_arr.sum())
    bin_arr = np.reshape(bin_arr, (1,size))
    return bin_arr


def expand(im, filter_vec):
    """
    a helper method for expanding the image by double from it's input size
    :param im: the input picture to expand
    :param filter_vec: a custom filter in case we'd like to convolve with different one
    :return: the expanded picture after convolution
    """
    new_expand = np.zeros(shape=(int(im.shape[0]*2), int(im.shape[1]*2)))
    new_expand[::2,::2] = im
    new_expand = scipy.signal.convolve2d(new_expand, 2*filter_vec, mode='same')
    new_expand = scipy.signal.convolve2d(new_expand, np.transpose(2*filter_vec), mode='same')

    return new_expand


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    a method for building a gaussian pyramid
    :param im: the input image to construct the pyramid from
    :param max_levels: maximum levels in the pyramid
    :param filter_size: the size of the gaussian filter we're using
    :return: an array representing the pyramid
    """
    filter_vec = create_gaussian_line(filter_size)
    # creating duplicate for confy use
    temp_im = im
    pyr = [im]
    kernel = np.array([0.0625,0.25,0.375,0.25,0.0625])
    kernel = kernel.reshape((1,5))
    for i in range(max_levels - 1):
        # blurring the cur layer
        #temp_im_temp = cv2.filter2D(temp_im,-1,kernel,borderType=cv2.B)
        temp_im = scipy.signal.convolve2d(temp_im, filter_vec, mode='same')
        temp_im = scipy.signal.convolve2d(temp_im, np.transpose(filter_vec), mode='same')
        # sampling only every 2nd row and column
        temp_im = temp_im[::2, ::2]
        pyr.append(temp_im)

    return pyr, filter_vec

def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    a method for building a laplacian pyramid
    :param im: the input image to construct the pyramid from
    :param max_levels: maximum levels in the pyramid
    :param filter_size: the size of the laplacian filter we're using
    :return: an array representing the pyramid
    """
    pyr = []
    org_reduce, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    for i in range(max_levels - 1):
        temp_expand = expand(org_reduce[i + 1], filter_vec)
        org_layer = org_reduce[i]
        temp = org_layer - temp_expand
        pyr.append(temp)
    # plt.imshow(org_reduce[-1], cmap='gray')
    # plt.show()
    pyr.append(org_reduce[-1])
    return pyr, filter_vec

def normal_high_frequency(imglayer):
    '''
    TODO:将残差进行归一化处理
    :param imglayer:
    :return:
    '''
    imglayer_no = np.zeros((imglayer.shape[0],imglayer.shape[1]))
    img_min = imglayer.min()
    img_max = imglayer.max()
    eof = 1e-6

    if img_max > eof and img_min < eof:
        imglayer_no[imglayer > eof] = imglayer[imglayer > eof] / (img_max + eof)
        imglayer_no[imglayer < eof] = imglayer[imglayer < eof] / (img_min + eof)
        imglayer_no[imglayer - eof < eof] = 0
    elif img_min >= eof:
        imglayer_no = imglayer / img_max
    elif img_max < eof:
        imglayer_no = imglayer / img_min

    return imglayer_no

def enhance_1(imgnormal,p1,p2,alpha):
    '''
    TODO:对归一化后的残差图进行第一次增强
    :param imgnormal:
    :param p:
    :return:
    '''
    img_expand = np.exp(-np.power(imgnormal, 2)/p1) + 1
    mean = np.mean(img_expand)
    img_expand = img_expand - mean
    eof = 1e-6

    coeff = np.zeros((imgnormal.shape[0], imgnormal.shape[1]))
    coeff = alpha*(img_expand/(abs(img_expand)+eof))*np.power(abs(img_expand),p2)

    return  coeff






def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    TODO：含增强效果的拉普拉斯重构，增强因子coeff
    a method that constructs the original image from it's laplacian pyramid
    :param lpyr: the laplacian pyramid of the image we'd like to construct
    :param filter_vec: the filter vector to be used in the reconstruction
    :param coeff: the coefficients for each layer of the pyramid
    :return: the reconstructed image
    """

    pyr_updated = np.multiply(lpyr, coeff)
    # wl = [3.0,3.0,3.0,1.0,1.0,1.0]

    cur_layer = lpyr[-1]
    for i in range(len(pyr_updated) - 2, -1, -1):
        cur_layer = expand(cur_layer, filter_vec) + pyr_updated[i]
    cur_layer[cur_layer<0]=0.0
    return cur_layer


def render_pyramid(pyr, levels):
    """
    render the pyramid and construct a single image representing all the layers horizontally
    :param pyr: the image's pyramid
    :param levels: the number of levels of the pyramid
    :return: an image representing all the pyramid's layers horizontally
    """
    positionLst = []
    finalLst = []
    if levels > len(pyr):
        print("error. number of levels to display is more than max_levels")
    width = 0

    for i in range(levels):
        # streching each layer
        pyr[i] = strech_helper(pyr[i])
        width += pyr[i].shape[1]
        positionLst.append((pyr[i].shape[0], pyr[i].shape[1]))

    for i in range(levels):
        zeros = np.zeros(shape=(pyr[0].shape[0], pyr[i].shape[1]))
        zeros[:positionLst[i][0], :positionLst[i][1]] = pyr[i]
        finalLst.append(zeros)
    res = np.concatenate(finalLst, axis=1)
    return res

def strech_helper(im):
    """
    helper function for streching and equalizing the image between 0 and 1
    :param im: input picture to equalize
    :return: equalized picture between 0 and 1
    """
    return (im - np.min(im))/(np.max(im) - np.min(im))


def display_pyramid(pyr, levels):
    """
    displaying the pyramid into the screen using plt
    :param pyr: the input pyramid to be displayed
    :param levels: number of levels of the pyramid
    """
    res = render_pyramid(pyr, levels)
    plt.imshow(res, cmap='bone')
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    a method for blending 2 pictures using a binary mask
    :param im1: the first picture to blend
    :param im2: the second picture to blend
    :param mask: the binary mask
    :param max_levels: number of max levels to be used while constructing the pyramids
    :param filter_size_im: size of the filter for the images
    :param filter_size_mask: size of the filter for the mask
    :return:
    """
    mask = mask.astype(np.float64)
    lap_pyr1, filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    lap_pyr2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)[0]
    gauss_pyr = build_gaussian_pyramid(mask, max_levels, filter_size_mask)[0]
    # TODO: find more elegant way instead of loop
    for i in range(len(gauss_pyr)):
        gauss_pyr[i] = np.array(gauss_pyr[i], dtype=np.float64)
    new_lap_pyr = []
    coeff = [1] * max_levels
    for i in range(max_levels):
        cur_lap_layer = np.multiply(gauss_pyr[i], lap_pyr1[i]) + np.multiply(1 - gauss_pyr[i], lap_pyr2[i])
        new_lap_pyr.append(cur_lap_layer)
    final_image = laplacian_to_image(new_lap_pyr, filter_vec, coeff)
    return np.clip(final_image, 0, 1)

def relpath(filename):
    """
    helper method for using relative paths to load the pictures
    :param filename: the relative path to be parsed
    :return: the real path
    """
    return os.path.join(os.path.dirname(__file__), filename)

def blending_example1():
    """
    a method for creating a blending example constructing a blend from 2 rgb images
    :return: the blended picture
    """
    pic_desert = read_image(relpath("./externals/pic_desert.jpg"), 2)
    pic_pool = read_image(relpath("./externals/pic_swim.jpg"), 2)
    mask = read_image(relpath("./externals/mask_desert.jpg"), 1)
    # making the mask binary (normalizing 2 original values)
    mask = strech_helper(mask).astype(np.bool)
    print(pic_desert.shape[2])
    [R1, G1, B1] = np.dsplit(pic_desert, pic_desert.shape[2])
    [R2, G2, B2] = np.dsplit(pic_pool, pic_pool.shape[2])
    R1 = np.reshape(R1, (512,1024))
    R2 = np.reshape(R2, (512,1024))
    G1 = np.reshape(G1, (512,1024))
    G2 = np.reshape(G2, (512,1024))
    B1 = np.reshape(B1, (512,1024))
    B2 = np.reshape(B2, (512,1024))

    blend1 = pyramid_blending(R2, R1, mask, 3, 3, 3)
    blend2 = pyramid_blending(G2, G1, mask, 3, 3, 3)
    blend3 = pyramid_blending(B2, B1, mask, 3, 3, 3)

    blend1 = np.reshape(blend1, (blend1.shape[0], blend1.shape[1], 1))
    blend2 = np.reshape(blend2, (blend2.shape[0], blend3.shape[1], 1))
    blend3 = np.reshape(blend3, (blend3.shape[0], blend3.shape[1], 1))

    new_pic = np.concatenate((blend1, blend2, blend3), axis=2)
    # plotting the images
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    ax1.imshow(pic_desert)
    ax2.imshow(pic_pool)
    ax3.imshow(mask, cmap='gray')
    ax4.imshow(new_pic)
    plt.show()

    return pic_desert, pic_pool, mask, new_pic

def blending_example2():
    """
    a method for creating a blending example constructing a blend from 2 rgb images
    :return: the blended picture
    """
    pic_earth = read_image(relpath("./externals/pic_earth.jpg"), 2)
    pic_asteroid = read_image(relpath("./externals/pic_asteroid.jpg"), 2)
    mask = read_image(relpath("./externals/mask_asteroid.jpg"), 1)
    # making the mask binary (normalizing 2 original values)
    mask = strech_helper(mask).astype(np.bool)
    [R1, G1, B1] = np.dsplit(pic_earth, pic_earth.shape[2])
    [R2, G2, B2] = np.dsplit(pic_asteroid, pic_asteroid.shape[2])
    R1 = np.reshape(R1, (1024,1024))
    R2 = np.reshape(R2, (1024,1024))
    G1 = np.reshape(G1, (1024,1024))
    G2 = np.reshape(G2, (1024,1024))
    B1 = np.reshape(B1, (1024,1024))
    B2 = np.reshape(B2, (1024,1024))

    blend1 = pyramid_blending(R2, R1, mask, 3, 3, 3)
    blend2 = pyramid_blending(G2, G1, mask, 3, 3, 3)
    blend3 = pyramid_blending(B2, B1, mask, 3, 3, 3)

    blend1 = np.reshape(blend1, (blend1.shape[0], blend1.shape[1], 1))
    blend2 = np.reshape(blend2, (blend2.shape[0], blend3.shape[1], 1))
    blend3 = np.reshape(blend3, (blend3.shape[0], blend3.shape[1], 1))

    new_pic = np.concatenate((blend1, blend2, blend3), axis=2)
    # plotting the images
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    ax1.imshow(pic_earth)
    ax2.imshow(pic_asteroid)
    ax3.imshow(mask, cmap='gray')
    ax4.imshow(new_pic)
    plt.show()

    return pic_earth, pic_asteroid, mask, new_pic


# def savepng(img,filepath):
#     img = (65535*(img - img.min())/img.ptp()).astype(np.uint16)
#
#     with open(filepath,'wb') as f:
#         writer = png.Writer(width= img.shape[0], height= img.shape[1],bitdepth=16,greyscale=True)
#         img = img.tolist()
#         writer.write(f,img)
#
#
# if __name__ =='__main__':
#     img1 = cv2.imread('/home/chenhao/device/method_test_save/grayAd/ori_retag.png',
#                      cv2.IMREAD_ANYDEPTH)
#     img1 = cv2.resize(img1,(2048,2048),cv2.INTER_LANCZOS4)
#     img2 = cv2.imread('/home/chenhao/device/method_test_save/grayAd/pro.png',
#                       cv2.IMREAD_UNCHANGED)
#     img2 = cv2.resize(img2,(2048,2048),cv2.INTER_LANCZOS4)
#     max_levels = 8
#     filter_size_im = 5
#     #coeff = 7.0
#     imp1, vec1 = build_laplacian_pyramid(img1, max_levels, filter_size_im)
#     imp2, vec2 = build_laplacian_pyramid(img2, max_levels,filter_size_im)
#     # imp_0 = normal_high_frequency(imp[0])
#     #coeff = enhance_1(imp_0,0.3,0.75,5)
#     #imp_0 = coeff*imp[0]
#     #savepng(imp_0,'F:/test_pyramid_0_ch.png')
#     # for i in range(max_levels):
#     #     name = 'F:/test_pyramid_'+str(i)+'.png'
#     #     savepng(imp[i],name)
#     imp2[7] = imp1[7]
#     temp = laplacian_to_image(imp2,vec2,[1.0]*len(imp2))
#     temp = temp.astype(np.uint16)
#     temp = cv2.resize(temp,(3032,3023),cv2.INTER_LANCZOS4)
#     import numpngw
#     numpngw.write_png('/home/chenhao/xiaoxifei.png',temp)

    #print('finished')

# def getimg(filename):
#     if filename.endswith('.dcm'):
#         dcm = pydicom.read_file(filename)
#         img = dcm.pixel_array
#     elif filename.endswith('.png'):
#         img = cv2.imread(filename,cv2.IMREAD_UNCHANGED)
#     else:
#         img = cv2.imread(filename)
#     return img

def needResize(img):
    '''
    TODO: 判断输入的图像大小是否需要进行变化，用于进行金子塔获取的图像为了防止出现获得奇数大小的子图需要预先进行判断，
    如果有需要，按照就近原则k×2^n 来进行调整
    :param img:
    :return:
    '''
    flag = True
    row,col = img.shape
    # 将大小转为二进制数
    binrow = bin(row)
    bincol = bin(col)
    # 记录二进制数位数
    lengthrow = len(binrow) - 3
    lengthcol = len(bincol) - 3

    # 判断是否需要调整,flag 为false不需要调整
    larger_row = 2**lengthrow
    larger_col = 2**lengthcol

    smaller_row = 2**(lengthrow-1)
    smaller_col = 2**(lengthcol-1)

    new_row =row
    new_col = col
    if (row % larger_row == 0 and col % larger_col == 0) or (row % smaller_row == 0 and col % smaller_col == 0):
        flag = False
        return img,new_row,new_col,flag
    else:
        # 如果需要调整，求取调整后最佳值
        flag = True
        flag_row = True
        flag_col = True
        if row % larger_row ==0 or row % smaller_row ==0:   # row不需要进行调整
            flag_row = False
            new_row = row
        if col % larger_col ==0 or col % smaller_col ==0:   # col不需要进行调整
            flag_col = False
            new_col = col

        if flag_row:   # 需要进行row的调整
            temp = []
            for i in range(4):  # row的分布只可能位于4个数之间
                temp.append(abs(row-smaller_row*(i+1)))
            index = temp.index(min(temp))

            new_row = smaller_row*(index+1)

        if flag_col:   # 需要进行col的调整
            temp = []
            for i in range(4):
                temp.append(abs(col-smaller_col*(i+1)))
            index = temp.index(min(temp))

            new_col = smaller_col*(index+1)

        return img,new_row,new_col,flag


def img_adjust_ribsuppression(img,imgpro,levels=8,filter_size=5):
    '''
    TODO: use for image gray adjust after ribsuppression
    :param img: ndarray img input no rib suppression
    :param imgpro: ndarray img after ribsuppression
    :return img_ad: ndarray img after gray adjust
    '''

    row,col = img.shape
    if row % 2**levels !=0 or col % 2**levels !=0:
        raise Exception('img size and levels do not match')

    new_row,new_col,flag = needResize(img)

    # if img size should be resize
    if flag:
        img = cv2.resize(img,(new_row,new_col),cv2.INTER_LANCZOS4)
        imgpro = cv2.resize(imgpro,(new_row,new_col),cv2.INTER_LANCZOS4)

    # build pyramid
    L_img,vec_img = build_laplacian_pyramid(img,levels,filter_size)
    L_imgpro,vec_pro = build_laplacian_pyramid(imgpro,levels,filter_size)

    # replace L_imgpro`s last level
    L_imgpro[-1] = L_img[-1]

    # reconstruct imgpro
    imgpro_new = laplacian_to_image(L_imgpro, vec_pro, [1.0] * len(L_imgpro))
    imgpro_new = imgpro_new.astype(np.uint16)

    if flag:
        imgpro_new = cv2.resize(imgpro_new,(row,col),cv2.INTER_LANCZOS4)

    return  imgpro_new





def img_blend(img1, img2, mask, level=5, filter_size=5, keep_size=True):
    '''
    TODO: 用来进行图像融合
    :param img1:
    :param img2:
    :param mask:
    :param level:
    :param filter_size:
    :return:
    '''
    row, col = img1.shape
    mask = cv2.resize(mask, (col, row))
    # if row % 2 ** level != 0 or col % 2 ** level != 0:
    #     raise Exception('img size and levels do not match')

    _, new_row, new_col, flag = needResize(img1)

    if flag:
        img1 = cv2.resize(img1, (new_row, new_col), cv2.INTER_LANCZOS4)
        img2 = cv2.resize(img2, (new_row, new_col), cv2.INTER_LANCZOS4)
        mask = cv2.resize(mask, (new_row, new_col))

    l1Pyr, filterVec1 = build_laplacian_pyramid(img1, level, filter_size)
    l2Pyr, filterVec2 = build_laplacian_pyramid(img2, level, filter_size)
    maskInFlots = mask.astype(np.float32)
    gaussMaskPyr, filterVec3 = build_gaussian_pyramid(maskInFlots, level,
                                                      filter_size)
    lOut = []
    lenOfPyr = len(l1Pyr)
    for i in range(lenOfPyr):
        lOut.append(np.multiply(gaussMaskPyr[i], l1Pyr[i]) +
                    (np.multiply(1 - gaussMaskPyr[i], l2Pyr[i])))
    blendedIm = laplacian_to_image(lOut, filterVec1, [1] * lenOfPyr)
    # blendedImClip = np.clip(blendedIm, 0, 1)

    if keep_size:
        blendedIm = cv2.resize(blendedIm, (col, row), cv2.INTER_LANCZOS4)
    return blendedIm.astype(np.uint16)






















# def grayAdPy(file1,file2,max_level=8,filter_size=5):
#     '''
#
#     :param file1: 处理前的图像
#     :param file2: 处理后的图像
#     :param max_level: 金字塔层数
#     :param filter_size: 滤波器大小
#     :return:
#     '''
#     img1 = getimg(file1)
#     img2 = getimg(file2)
#     row,col = img2.shape
#     img1,row1,col1,flag1 = needResize(img1)
#     img2,row2,col2,flag2 = needResize(img2)
#
#     if flag1 :
#         img1 = cv2.resize(img1,(row1,col1),cv2.INTER_LANCZOS4)
#     if flag2 :
#         img2 = cv2.resize(img2,(row2,col2),cv2.INTER_LANCZOS4)
#
#     L1,vec1 = build_laplacian_pyramid(img1,max_level,filter_size=filter_size)
#     L2,vec2 = build_laplacian_pyramid(img2,max_level,filter_size=filter_size)
#
#     L2[-1] = L1[-1]
#
#     img_re = laplacian_to_image(L2,vec2,[1.0]*len(L2))
#
#     if flag2:
#         img_re = cv2.resize(img_re,(row,col),cv2.INTER_LANCZOS4)
#
#     img_re = img_re.astype(np.uint16)
#
#     return img_re
#
# def getfile(path):
#     '''
#     TODO: 用于返回专门用于grayAd需要的file list
#     :param path:
#     :return:
#     '''
#
#     list1 = list(map(lambda x : path+'withbone/' + x, os.listdir(path + 'withbone')))
#     list2 = list(map(lambda x : path+'soft/' + x, os.listdir(path + 'soft')))
#     length = len(list1)
#
#     return list1,list2,length



# 进行算法调试与测试
# if __name__ =='__main__':
#     # img1是参考图 img2是待处理的图
#     img1 = cv2.imread('/home/chenhao/device/method_test_save/grayAd/withbone_png/1.png',cv2.IMREAD_UNCHANGED)
#     img2 = cv2.imread('/home/chenhao/device/method_test_save/grayAd/all_bone_png/1.png',cv2.IMREAD_UNCHANGED)
#     max_levels = 8
#     filter_size_im = 5
#     img1 = cv2.resize(img1,(2048,2048),cv2.INTER_LANCZOS4)
#     img2 = cv2.resize(img2,(2048,2048),cv2.INTER_LANCZOS4)
#
#     L1,vec1 = build_laplacian_pyramid(img1,max_levels,filter_size_im)
#     L2,vec2 = build_laplacian_pyramid(img2,max_levels,filter_size_im)
#
#     #img11 = laplacian_to_image(L1,vec1,[1.0]*len(L1))
#
#     L2[-1] = L1[-1]
#
#     img = laplacian_to_image(L2,vec2,[1.0]*len(L2))
#
#     img = img.astype(np.uint16)
#     import numpngw
#     numpngw.write_png('/home/chenhao/device/method_test_save/grayAd/bone_Lp.png',img)



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


if __name__ =="__main__":
    path1 = r'G:\Data\DicomImages\train2048\soft_remove_tag'
    path2 = r'G:\Data\DicomImages\train2048\src\data'
    maskpath = r'G:\Data\DicomImages\train2048\mask\data'

    savepath = r'G:\Data\DicomImages\train2048\test_blend'

    lists = os.listdir(path1)
    for file in lists:
        print('{} is processed'.format(file))
        dcm1 = pydicom.read_file(os.path.join(path1, file))
        img1 = dcm1.pixel_array
        dcm2 = pydicom.read_file(os.path.join(path2, file))
        img2 = removal_tag(dcm2)


        mask = cv2.imread(os.path.join(maskpath, file.split('.dcm')[0] + '.png'), cv2.IMREAD_UNCHANGED)
        if mask is None:

            raise Exception("np mask input")
        mask = enlarge_mask(mask, 30)
        img_new = img_blend(img1, img2, mask)

        dcm1.PixelData = img_new.tobytes()
        dcm1.save_as(os.path.join(savepath, file))










