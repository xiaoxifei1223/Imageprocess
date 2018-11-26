
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread as imread
from skimage.color import rgb2gray
from scipy import signal as sig
from scipy.ndimage.filters import convolve
import os
import pydicom
import cv2
import numpngw

IDENTITY_KERNEL_SIZE = 1
BINOMIAL_MAT = [0.5, 0.5]
GRAY = 1
RGB = 2
NORM_PIX_FACTOR = 255
ROWS = 0
COLS = 1
LARGEST_IM_INDEX = 0
DIM_RGB = 3


def read_image(filename, representation):
    """this function reads a given image file and converts it into a given
    representation:
    filename - string containing the image filename to read.
    representation - representation code, either 1 or 2 defining if the
                     output should be either a grayscale image (1) or an
                     RGB image (2).
    output - the image in the given representation when the pixels are
             of type np.float32 and normalized"""
    filename = os.path.abspath(filename)
    if not os.path.exists(filename):
        return
    im = imread(filename)
    if im.dtype == np.float32:
        '''I don't handle this case, we asume imput in uint8 format'''
        return
    if representation == GRAY:
        im = rgb2gray(im).astype(np.float32)
        if np.max(im) > 1:
            '''not suppose to happened'''
            im /= NORM_PIX_FACTOR
        return im
    im = im.astype(np.float32)
    im /= NORM_PIX_FACTOR
    return im


def getGaussVec(kernel_size):
    '''
    gets the gaussian vector in the length of the kernel size
    :param kernel_size: the length of the wished kernel
    :return: the 1d vector we want
    '''
    if kernel_size == IDENTITY_KERNEL_SIZE:
        return [1]
    return sig.convolve(BINOMIAL_MAT, getGaussVec(kernel_size - 1)).astype(np.float32)


def getImAfterBlur(im, filter):
    '''
    return the image after row and col blur
    :param im: the image to blur
    :param filter: the filter to blur with
    :return: blurred image
    '''
    blurXIm = convolve(im, filter)
    blurIm = convolve(blurXIm, filter.transpose())
    return blurIm


def reduceIm(currIm, gaussFilter, filter_size):
    '''
    reduce an image
    :param currIm: the image to reduce by 4
    :param gaussFilter: the filter to blur with the image before reduce
    :param filter_size: the size of the filter
    :return: the reduced image
    '''
    blurIm = getImAfterBlur(currIm, gaussFilter)
    reducedImage = blurIm[::2, ::2]
    return reducedImage.astype(np.float32)


def expandIm(currIm, gaussFilterForExpand, filter_size):
    '''
    expand an image
    :rtype : np.float32
    :param currIm: the image to expand by 4
    :param gaussFilterForExpand: the filter to blur with the expand image
    :param filter_size: the size of the filter
    :return: an expand image
    '''
    expandImage = np.zeros((2 * currIm.shape[0], 2 * currIm.shape[1]))
    expandImage[::2, ::2] = currIm
    expandRes = getImAfterBlur(expandImage, gaussFilterForExpand)
    return expandRes.astype(np.float32)


def getNumInInPyr(im, max_levels):
    '''
    return maximum number of images in pyramid
    :param im: tne original image
    :param max_levels: an initial limitation
    :return: the real limitation
    '''
    numRows, numCols = im.shape[ROWS], im.shape[COLS]

    limRows = np.floor(np.log2(numRows)) - 3
    limCols = np.floor(np.log2(numCols)) - 3
    numImInPyr = np.uint8(np.min([max_levels, limCols, limRows]))
    return numImInPyr


def build_gaussian_pyramid(im, max_levels, filter_size):
    '''
    construct a Gaussian pyramid of a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the Gaussian filter
    :return: Gaussian pyramid as standard python array and the filter vec
    '''
    numImInPyr = getNumInInPyr(im, max_levels)
    gaussFilter = np.array(getGaussVec(filter_size)).reshape(1, filter_size)

    gaussPyr = [im]
    currIm = im
    for i in range(1, numImInPyr):
        currIm = reduceIm(currIm, gaussFilter, filter_size)
        gaussPyr.append(currIm)
    return gaussPyr, gaussFilter


def build_laplacian_pyramid(im, max_levels, filter_size):
    '''
    construct a Laplacian pyramid of a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the Gaussian filter
    :return: Laplacian pyramid as standard python array and the filter vec
    '''
    gaussFilter = np.array(getGaussVec(filter_size)).reshape(1, filter_size)
    laplacianPyr = []

    gaussPyr = build_gaussian_pyramid(im, max_levels, filter_size)[0]
    numImInPyr = len(gaussPyr)

    for i in range(numImInPyr - 1):
        laplacianPyr.append(gaussPyr[i] - expandIm(
            gaussPyr[i + 1], np.multiply(2, gaussFilter), filter_size))
    laplacianPyr.append(gaussPyr[numImInPyr - 1])
    return laplacianPyr, gaussFilter


def laplacian_to_image(lpyr, filter_vec, coeff):
    '''
    reconstruction of an image from its Laplacian Pyramid
    :param lpyr: Laplacian pyramid
    :param filter_vec: the filter that was used in order to
            construct the pyramid
    :param coeff: the coefficient of each image in the pyramid
    :return: reconstruction of an image from its Laplacian Pyramid
    '''
    numIm, numCoe = len(lpyr), len(coeff)
    if numIm != numCoe:
        '''invalid input'''
        return
    gni = lpyr[numIm - 1]
    for i in range(numIm - 1):
        gni = expandIm(gni, np.multiply(2, filter_vec), len(filter_vec)) + (
            lpyr[numIm - 1 - i - 1] * coeff[len(coeff) - 1 - i])
    return gni.astype(np.float32)


def strechIm(im, newMin, newMax):
    """
    strech the image to [newMin, newMax]
    :param newMax: max vlue to stretch to
    :param newMin: min value to stretch to
    :param im: float 32 image
    :return: stretched im
    """
    inMin, inMax = np.min(im), np.max(im)
    stretchedIm = (im - inMin) * ((newMax - newMin) / (inMax - inMin)) + newMin
    return stretchedIm


def render_pyramid(pyr, levels):
    '''
    creates a single black image in which the pyramid levels of the
        given pyramid pyr are stacked horizontally
    :param pyr: either a Gaussian or Laplacian pyramid
    :param levels: number of levels to present in the result
    :return: single black image in which the pyramid levels of the
        given pyramid pyr are stacked horizontally
    '''
    levels = min(levels, len(pyr))
    numRows = pyr[LARGEST_IM_INDEX].shape[ROWS]
    numCols = pyr[LARGEST_IM_INDEX].shape[COLS]
    for i in range(1, levels):
        numCols += pyr[i].shape[COLS]
    pyrIm = np.zeros([numRows, numCols])
    curPlace = 0
    for i in range(levels):
        stretchedIm = strechIm(pyr[i], 0, 1)
        rows = stretchedIm.shape[ROWS]
        cols = stretchedIm.shape[COLS]
        pyrIm[:rows, curPlace:curPlace + cols] = stretchedIm
        curPlace = curPlace + cols
    return pyrIm.astype(np.float32)


def display_pyramid(pyr, levels):
    '''
    display at max the amount of levels from the pyr
    :param pyr: either a Gaussian or Laplacian pyramid
    :param levels: number of levels to present in the result
    '''
    res = render_pyramid(pyr, levels)
    plt.figure()
    plt.imshow(res, cmap=plt.cm.gray)
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im,
                     filter_size_mask):
    '''
    blending images using pyramids.
    :rtype : tuple
    :param im1: first im to blend - grayscale
    :param im2: second im to blend - grayscale
    :param mask:  is a boolean (i.e. dtype == np.bool) mask containing
        True and False representing which parts of im1 and im2 should
        appear in the resulting im_blend.
    :param max_levels:  is the max_levels parameter you should use when
            generating the Gaussian and Laplacian pyramids
    :param filter_size_im:  is the size of the Gaussian filter
            (an odd scalar that represents a squared filter) which defining the
            filter used in the construction of the Laplacian pyramids of
            im1 and im2
    :param filter_size_mask: is the size of the Gaussian filter(an odd scalar
            that represents a squared filter) which defining the filter used
            in the construction of the Gaussian pyramid of mask.
    :return: blended image using laplacian and gausian pyramids.
    '''
    l1Pyr, filterVec1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    l2Pyr, filterVec2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    maskInFlots = mask.astype(np.float32)
    gaussMaskPyr, filterVec3 = build_gaussian_pyramid(maskInFlots, max_levels,
                                                      filter_size_mask)
    lOut = []
    lenOfPyr = len(l1Pyr)
    for i in range(lenOfPyr):
        lOut.append(np.multiply(gaussMaskPyr[i], l1Pyr[i]) +
                    (np.multiply(1 - gaussMaskPyr[i], l2Pyr[i])))
    blendedIm = laplacian_to_image(lOut, filterVec1, [1] * lenOfPyr)
    # blendedImClip = np.clip(blendedIm, 0, 1)
    return blendedIm.astype(np.uint16)


def relpath(filename):
    '''
    return the real path of filename
    :param filename: relative path name
    :return: the real path of filename
    '''
    return os.path.join(os.path.dirname(__file__), filename)


def generateFigure(im1, im2, mask, blendedIm):
    '''
    generate the figure with all the necessary images.
    :param im1:
    :param im2:
    :param mask:
    :param blendedIm:
    '''
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(im1)
    plt.subplot(2, 2, 2)
    plt.imshow(im2)
    plt.subplot(2, 2, 3)
    plt.imshow(mask, cmap=plt.cm.gray)
    plt.subplot(2, 2, 4)
    plt.imshow(blendedIm)
    plt.show()


def doMyBlend(im1Path, im2Path, maskPath):
    '''
    performing RGB pyramid blending
    :param im1Path: relative path of im1
    :param im2Path: relative path of im2
    :param maskPath: relative path of mask (binary image)
    :return: im1, im2, mask, blendedIm
    '''
    max_levels = 5
    filter_size_im = 5
    filter_size_mask = 5
    mask32 = read_image(relpath(maskPath), 1)
    mask = mask32.astype(np.bool)
    im1 = read_image(relpath(im1Path), 2)
    im2 = read_image(relpath(im2Path), 2)

    blendedIm = np.zeros((im1.shape[0], im1.shape[1], im1.shape[2])).\
        astype(np.float32)
    for i in range(DIM_RGB):
        blendedIm[:, :, i] = pyramid_blending(im1[:, :, i], im2[:, :, i],
                                              mask, max_levels, filter_size_im,
                                              filter_size_mask)

    generateFigure(im1, im2, mask, blendedIm)
    return im1, im2, mask, blendedIm


def blending_example1():
    '''
    performing RGB pyramid blending
    :return: im1, im2, mask, blendedIm
    '''
    return doMyBlend('external/bonus/plate.jpg', 'external/bonus/rabbit.jpg',
                     'external/bonus/maskRabbit_converted.jpg')


def blending_example2():
    '''
    performing RGB pyramid blending
    :return: im1, im2, mask, blendedIm
    '''
    return doMyBlend('external/bonus/camel.jpg', 'external/bonus/view.jpg',
                     'external/bonus/maskCamelBool_converted.jpg')


def testBlend():
    max_levels = 5
    filter_size_im = 5
    filter_size_mask = 5
    # mask32 = read_image('external/bonus/maskCamelBool.jpg', 1)
    # mask = mask32.astype(np.bool)
    # im1 = read_image('external/bonus/camel.jpg', 1)
    # im2 = read_image('external/bonus/view.jpg', 1)
    #im1 = pydicom.read_file('/home/chenhao/device/method_test_save/Crop/src/307-20180614119.dcm').pixel_array
    im2 = cv2.imread('/home/chenhao/device/method_test_save/Crop/src/307-20180614119.png',cv2.IMREAD_UNCHANGED)
    im2 = im2.astype(np.float32)
    #im2 = pydicom.read_file('/home/chenhao/device/method_test_save/Crop/soft/307-20180614119.dcm').pixel_array
    im1 = cv2.imread('/home/chenhao/device/method_test_save/Crop/soft/307-20180614119.png', cv2.IMREAD_UNCHANGED)
    im1 = im1.astype(np.float32)
    mask = cv2.imread('/home/chenhao/device/method_test_save/Crop/mask/307-20180614119.png',cv2.IMREAD_UNCHANGED)
    mask = cv2.resize(mask, (2048, 2048))
    mask = mask.astype(np.bool)
    im_blend = pyramid_blending(im1, im2, mask, max_levels,
                                      filter_size_im, filter_size_mask)



    plt.figure()
    plt.imshow(im_blend, cmap=plt.cm.gray)
    plt.show(block=True)

if __name__ =="__main__":
    testBlend()