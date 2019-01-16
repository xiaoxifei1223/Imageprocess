# coding:utf-8
# TODO: 用来进行肺部肋骨边缘的增广
import numpy as np
import cv2
import scipy.signal as signal
import pydicom

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
        bin_arr = signal.convolve(bin_arr, org_arr)
    bin_arr = np.divide(bin_arr, bin_arr.sum())
    bin_arr = np.reshape(bin_arr, (1,size))
    return bin_arr

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
        temp_im = signal.convolve2d(temp_im, filter_vec, mode='same')
        temp_im = signal.convolve2d(temp_im, np.transpose(filter_vec), mode='same')
        # sampling only every 2nd row and column
        temp_im = temp_im[::2, ::2]
        pyr.append(temp_im)

    return pyr, filter_vec

def expand(im, filter_vec):
    """
    a helper method for expanding the image by double from it's input size
    :param im: the input picture to expand
    :param filter_vec: a custom filter in case we'd like to convolve with different one
    :return: the expanded picture after convolution
    """
    new_expand = np.zeros(shape=(int(im.shape[0]*2), int(im.shape[1]*2)))
    new_expand[::2,::2] = im
    new_expand = signal.convolve2d(new_expand, 2*filter_vec, mode='same')
    new_expand = signal.convolve2d(new_expand, np.transpose(2*filter_vec), mode='same')

    return new_expand

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




def ribedge_aug(src):
   # new ribedge_aug
   if np.random.uniform(0,1) < 0.3:
       l1,v1 = build_laplacian_pyramid(src,5,5)
       coeff = [np.random.uniform(1.0,9.0),np.random.uniform(2.0,9.0)/2, 1.0,1.0,1.0]
       src_new = laplacian_to_image(l1,v1,coeff)
   else:
       src_new = src
   return src_new


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

    It = np.random.randint(5, 20)
    alpha_sum = 0
    v = 0.4
    if np.random.uniform(0, 1) > 0.5:
        n = np.random.uniform(v, 1.0 / v)
        alpha = np.random.uniform(0.01, 1.0)
        newsoft = alpha * np.power(soft_norm, n)
        alpha_sum += alpha
        for i in range(It - 1):
            n = np.random.uniform(v, 1.0 / v)
            alpha = np.random.uniform(0.01, 1.0)
            newsoft += alpha * np.power(soft_norm, n)
            alpha_sum += alpha

        newsoft = newsoft / alpha_sum

        newsoft = newsoft * (vmax - vmin) + vmin
        newsoft[soft == 0] = 0
    else:
        newsoft = soft


    return newsoft, (vmin, vmax)


def generator(bone, soft, mask, threshold):
    # bone = crop_bone_body(bone, inverse_mask)
    ws = np.random.uniform(0.02, 0.98)
    wb = 1 - ws

    # bone = inverse_bone(bone)
    fsoft, vthreshold = nonlinearImg(soft, mask)

    newsrc = (bone * (wb) + fsoft * (ws))

    return newsrc


if __name__ =="__main__":
    img = cv2.imread(r"G:\method_test_save\Rib_Aug\X15209072_bone.png",
                     cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(r'G:\method_test_save\Rib_Aug\X15209072.png',
                      cv2.IMREAD_UNCHANGED)
    rib_mask = cv2.imread(r"G:\method_test_save\Rib_Aug\ribedge_mask\X15209072.png",
                          cv2.IMREAD_UNCHANGED)
    soft = cv2.imread(r'G:\method_test_save\Rib_Aug\X15209072_soft.png',
                      cv2.IMREAD_UNCHANGED)

    img = cv2.resize(img, (2048, 2048))
    mask = cv2.resize(mask, (2048, 2048))
    soft = cv2.resize(soft, (2048, 2048))
    rib_mask = cv2.resize(rib_mask, (2048, 2048))
    count = 100
    for i in range(count):
        print(i)
        newsrc = generator(img,soft,mask,[0,1])
        img_rib = ribedge_aug(newsrc)
        img_rib = img_rib.astype(np.uint16)
        cv2.imwrite(r'G:\method_test_save\Rib_Aug\result\test_rib_'+str(i)+'.png', img_rib)
