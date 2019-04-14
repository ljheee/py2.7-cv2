# coding:utf-8
import cv2
import numpy as np


def hough(img, threshold=236): #250 一条,230,236两条
    thetas = np.deg2rad(np.arange(0, 180, 2))
    row, cols = img.shape;
    # 图片对角线长度
    diag_len = np.ceil(np.sqrt(row ** 2 + cols ** 2))
    # rhos = np.linspace(0, diag_len, int(diag_len))
    rhos = np.linspace(-diag_len, diag_len, int(2 * diag_len))
    cos_t = np.cos(thetas)
    sint_t = np.sin(thetas)
    num_thetas = len(thetas)

    # 构造计算矩阵
    vote = np.zeros((int(2 * diag_len), num_thetas), dtype=np.uint64)
    y_inx, x_inx = np.nonzero(img)

    # 计数
    for i in range(len(x_inx)):
        x = x_inx[i]
        y = y_inx[i]
        for j in range(num_thetas):
            rho = round(x * cos_t[j] + y * sint_t[j]) + diag_len
            vote[int(rho), j] += 1

    # 拿到所有大于阈值的 rhos theta
    indeies = np.argwhere(vote > threshold)
    rhos_idx = indeies[:, 0]
    theta_idx = indeies[:, 1]
    return vote, rhos[rhos_idx], thetas[theta_idx]


img = cv2.imread('hc.jpg')
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(grey, 70, 140)
cv2.imshow("edges", edges)

vote, rhos, thetas = hough(edges)
print 'finish hough'

vote = np.uint8(vote.T)
cv2.imshow("vote", vote)

for rho, theta in zip(rhos, thetas):
    x_center = img.shape[1] / 2
    x1 = int(x_center + 250)
    x2 = int(x_center - 250)
    print x1, x2, img.shape
    a = np.sin(theta)
    b = np.cos(theta)
    if a == 0.0:
        a = 1e-5
    if b == 0.0:
        b = 1e-5
    y1 = int((rho - x1 * b) / a)
    y2 = int((rho - x2 * b) / a)
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imshow("hough", img)
print 'finish'
cv2.waitKey(0)
