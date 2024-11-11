#!/usr/bin/env python
# encoding=gbk
import cv2
import numpy as np
from matplotlib import pyplot as plt

# ��ȡ�Ҷ�ͼ��
img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# �Ҷ�ͼ��ֱ��ͼ���⻯
dst = cv2.equalizeHist(gray)

# ֱ��ͼ
hist = cv2.calcHist([dst],[0],None,[256],[0,256])

plt.figure()
plt.title("gray")
plt.hist(dst.ravel(), 256)
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)

# ��ɫͼ��ֱ��ͼ���⻯

# ��ɫͼ����⻯,��Ҫ�ֽ�ͨ�� ��ÿһ��ͨ�����⻯
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)

plt.figure()
plt.title("bgr")
plt.hist(bH.ravel(),256, alpha=0.5, color='blue', label='Blue')
plt.hist(gH.ravel(),256, alpha=0.5, color='green', label='Green')
plt.hist(rH.ravel(),256, alpha=0.5,color='red', label='Red')
plt.show()
# �ϲ�ÿһ��ͨ��
result = cv2.merge((bH, gH, rH))
cv2.imshow("dst_rgb", np.hstack([img, result]))

cv2.waitKey(0)

