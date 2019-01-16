# coding:utf-8
# TODO: use traditional method to do object detection
import  cv2


def otsu_seg(img):

    ret_th, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return ret_th, bin_img

def find_pole(bin_img):
    img, contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = 0
    for i in range(len(contours)):
        area += cv2.contourArea(contours[i])
    area_mean = area / len(contours)
    mark = []
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) < area_mean:
            mark.append(i)

    return img, contours, hierarchy, mark

def draw_box(img,contours):
    img = cv2.rectangle(img,
                  (contours[0][0], contours[0][1]),
                  (contours[1][0], contours[1][1]),
                  (255,255,255),
                  3)
    return img



def main(img):
    ret, th = otsu_seg(img)
    img_new, contours, hierarchy, mark = find_pole(th)
    for i in range(len(contours)):
        if i not in mark:
            left_point = contours[i].min(axis=1).min(axis=0)
            right_point = contours[i].max(axis=1).max(axis=0)
            img = draw_box(img, (left_point, right_point))
    return img


if __name__ =="__main__":
    img = cv2.imread('G:/test.png')
    img = main(img)
    cv2.imwrite('G:/test_d.png', img)