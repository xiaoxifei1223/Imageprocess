import cv2
import source.cv_functions as cvf
import numpy as np
import os
from multiprocessing import Pool


saveflag = False


def GetRoiBoxes(filename, img, imgwidth,outpath):
    origin = img.copy()
    # rgb 转 gray
    try:
        gray = cvf.rgb2gray(rgb=origin)
    except:  # if origin image is grayscale
        gray = origin

    #  原图图像尺寸
    h, w = gray.shape

    #尺寸调整
    resized = cvf.resizing(image=gray, width=imgwidth)
    # 尺寸调整后的尺寸
    hr, wr = resized.shape
    cvf.save_image(saveflag, path=outpath, filename="resize_" + str(imgwidth)+"_"  +filename, image=resized)

    resized = cvf.bluring(binary_img=resized, k_size=11)

    mulmul = resized.copy()
    for i in range(20):

        # 将图像 按[0.3*均值，255]范围二值化
        ret, thresh = cv2.threshold(mulmul, np.average(mulmul) * 0.3, 255, cv2.THRESH_BINARY)
        cvf.save_image(saveflag,path=outpath, filename="thresh_" + str(imgwidth)+"_"  +str(i)+ "_" + filename, image=thresh)
        # 将图像 规范化到均值为127
        mulmul = cvf.normalizing(binary_img=resized * (thresh / 255))
        cvf.save_image(saveflag,path=outpath, filename="normal_" + str(imgwidth)+"_"  + str(i)+ "_" + filename, image=mulmul)


    movavg = cvf.moving_avg_filter(binary_img=mulmul, k_size=10)
    adap = cvf.adaptiveThresholding(binary_img=movavg, neighbor=111, blur=True, blur_size=3)
    cvf.save_image(saveflag,path=outpath, filename= "adaptive_"  + str(imgwidth)+"_" +filename,  image=255 - adap)


    masking = resized * ((255 - adap) / 255)
    cvf.save_image(saveflag,path=outpath, filename= "mask_" + str(imgwidth)+"_"  +filename, image=masking)


    movavg = cvf.moving_avg_filter(binary_img=masking, k_size=5)
    cvf.save_image(saveflag,path=outpath, filename="movavg_" + str(imgwidth)+"_"  +filename, image=movavg)

    ret, thresh = cv2.threshold(movavg, np.average(movavg) * 0.5, 255, cv2.THRESH_BINARY_INV)
    cvf.save_image(saveflag,path=outpath, filename= "thresh_" + str(imgwidth)+"_"  + str(imgwidth)+"_"  + filename, image=thresh)

    contours = cvf.contouring(binary_img=thresh)

    cv2.drawContours(resized, contours, -1, 0)
    cvf.save_image(saveflag, path=outpath, filename="Contours_" + str(imgwidth) + "_" + str(imgwidth) + "_" + filename,
                   image=resized)

    boxes_tmp = cvf.contour2box(contours=contours, padding=20)
    boxes = cvf.rid_repetition(boxes=boxes_tmp, binary_img=thresh)

    newboxes = []
    for box in boxes:
        box[0] = int( float(w) / float(wr) * box[0] )
        box[1] = int( float(h) / float(hr) * box[1] )
        box[2] = int( float(w) / float(wr) * box[2] )
        box[3] = int( float(h) / float(hr) * box[3] )
        newboxes.append(box)

    return newboxes




def GetLungROI(filename):

    PACK_PATH = "G:/Work/ChestRadioGraphy/tb/tb-voc/VOC2007/JPEGImages"
    outRoidir = "G:/Work/ChestRadioGraphy/tb/tb-voc/VOC2007/BoxJPEGImages"

    outpath = os.path.join(PACK_PATH,"ROI")
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    if not os.path.exists(outRoidir):
        os.makedirs(outRoidir)

    pyramid = [800, 500, 100]
    boxes = []
    if filename.split(".")[-1] == "jpg":
        # 导入图像
        img = cvf.load_image(path=os.path.join(PACK_PATH, filename))
        height, width = img.shape[0], img.shape[1]

        #  在不同下采样程度上的图像上计算ROI框
        for imgwidth in pyramid:
            boxespart = GetRoiBoxes(filename, img, imgwidth, outpath)
            boxes += boxespart

        # 抑制重叠的框，并选出最大的两个框
        boxes = cvf.NMS(boxes, (width, height), 0.5)

        # 对选出的两个框进行合理性判断以及 并框
        boxes = cvf.merge(boxes, (width, height))

        # 输出ROI区域
        RoiFilePath = os.path.join(outRoidir, filename + "_ROI.txt")
        with open(RoiFilePath, "w") as fp:
            if boxes != None:
                # 输出ROI区域
                fp.writelines("%d %d %d %d"%(boxes[0][0],boxes[0][1],boxes[0][0]+boxes[0][2],boxes[0][1]+boxes[0][3]) )

                for box in boxes:
                    rx, ry, rw, rh = box
                    cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
            else:
                # 没有正确找到ROI区域
                fp.writelines("None" + "\n")

        cvf.save_image(True, path=outpath, filename="boxes_" + filename, image=img)



if __name__ == '__main__':

    PACK_PATH = "G:/Work/ChestRadioGraphy/tb/tb-voc/VOC2007/JPEGImages"
    outRoidir = "G:/Work/ChestRadioGraphy/tb/tb-voc/VOC2007/BoxJPEGImages"


    filenames = os.listdir(PACK_PATH)
    pool = Pool(8)
    pool.map(GetLungROI, filenames)
    pool.close()
    pool.join()



