# coding:utf-8
import cv2
import numpy as np


def find_diff(left, right):
    '''输入两张图片'''
    output = left.copy()
    gray_left = np.int32(left)
    gray_right = np.int32(right)

    # 两张图片做减法
    minus_img = np.abs(gray_left - gray_right)
    minus_img = np.uint8(minus_img)
    minus_img_v = cv2.cvtColor(minus_img, cv2.COLOR_BGR2GRAY)

    # 二值化
    _, minus_img_v = cv2.threshold(minus_img_v, 21, 255, cv2.THRESH_BINARY)
    # 腐蚀&膨胀
    kernel = np.uint8(np.ones((3, 3)))
    minus_img_v = cv2.dilate(minus_img_v, kernel)
    minus_img_v = cv2.erode(minus_img_v, kernel)

    # 轮廓检查(找出最大亮区)
    contours, _ = cv2.findContours(minus_img_v, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 计算轮廓面积，只取最大的8个轮廓
    area_list = [cv2.contourArea(con) for con in contours]
    max7_contours_idx = np.argsort(area_list)[-8:]

    # 前5个区域，标识红色(作为结果)
    for idx in max7_contours_idx[-5:]:
        x, y, w, h = cv2.boundingRect(contours[idx])
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 3)

    # 后3个区域，标识绿色(作为备胎)
    for idx in max7_contours_idx[-8:-5]:
        x, y, w, h = cv2.boundingRect(contours[idx])
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
    return output


roi_y = 163
roi_x = 617
roi_height = 260
roi_width = 260
src = cv2.imread('src.jpg')  # 利用numpy数组切分设置ROI
cv2.imshow("src_left", src[roi_y:(roi_y + roi_height),
                       roi_x:(roi_x + roi_width)])
# https://blog.csdn.net/maweifei/article/details/53190690

left = cv2.imread('left.jpg')
right = cv2.imread('right.jpg')
output = find_diff(left, right)
cv2.imshow("output", output)
cv2.waitKey(0)
