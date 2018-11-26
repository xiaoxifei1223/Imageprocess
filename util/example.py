#!/usr/bin/env python3

"""
@license: Apache License Version 2.0
@author: Stefano Di Martino
Exact histogram matching
"""

from scipy import misc
from util.histogram_matching import ExactHistogramMatcher
import numpy as np
import png


def histogram_matching_rgb():
    target_img = misc.imread('F:/X15207198.png')
    reference_img = misc.imread('F:/307-1001059240_512.png')

    reference_histogram = ExactHistogramMatcher.get_histogram(reference_img)
    new_target_img = ExactHistogramMatcher.match_image_to_histogram(target_img, reference_histogram)
    misc.imsave('F:/rgb_out.png', new_target_img)


def histogram_matching_grey_values():
    target_img = misc.imread('/home/chenhao/device/method_test_save/grayAd/all_bone_png/1.png')    # 待处理图像
    reference_img = misc.imread('/home/chenhao/device/method_test_save/grayAd/withbone_png/1.png')  # 参考图像

    reference_histogram = ExactHistogramMatcher.get_histogram(reference_img)
    new_target_img = ExactHistogramMatcher.match_image_to_histogram(target_img, reference_histogram)
    new_target_img = new_target_img.astype(np.uint16)
    # misc.imsave('F:/grey_out.png', new_target_img)
    filename = '/home/chenhao/device/method_test_save/grayAd/oda_nobunaga.png'
    with open(filename, 'wb') as f:
        writer = png.Writer(width=new_target_img.shape[1], height=new_target_img.shape[0], bitdepth=16, greyscale=True)
        zgray2list = new_target_img.tolist()
        writer.write(f, zgray2list)


def main():
    # histogram_matching_rgb()
    histogram_matching_grey_values()


if __name__ == "__main__":
    main()

