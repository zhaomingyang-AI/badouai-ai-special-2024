# -*- coding: gbk -*-
# time: 2024/10/26 14:13
# file: sobel_laplace_canny.py
# author: flame
import cv2
from matplotlib import pyplot as plt

# ��ȡͼ���ļ� "lenna.png"������ 1 ��ʾ�Բ�ɫͼ����룬ͨ��˳��Ϊ BGR��
# ����ѡ�� 1 ����Ϊ������Ҫ����ͼ��Ĳ�ɫ��Ϣ���Ա��������ת��Ϊ�Ҷ�ͼ��
img = cv2.imread("lenna.png", 1)

# ������Ĳ�ɫͼ��ת��Ϊ�Ҷ�ͼ��
# cv2.COLOR_RGB2GRAY ������ʾ�� RGB ��ɫ�ռ�ת��Ϊ�Ҷ���ɫ�ռ䡣
# ת��Ϊ�Ҷ�ͼ����Ϊ�˼�ͼ�����ݣ����ټ��㸴�Ӷȣ����ұ�Ե����㷨ͨ���ڻҶ�ͼ����Ч�����á�
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# ʹ�� Sobel ������ˮƽ�����ϼ��ͼ���Ե��
# cv2.Sobel �������ڼ���ͼ���һ�׻���׵�����
# img_gray: ����ͼ�񣬱����ǵ�ͨ��ͼ��
# cv2.CV_64F: ���ͼ����������ͣ�����ѡ�� 64 λ�������Ա��������
# 1: x ����ĵ���������
# 0: y ����ĵ���������
# ksize: Sobel ���ӵľ���˴�С��3 ��ʾ 3x3 �ľ���ˡ�
img_sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)

# ʹ�� Sobel �����ڴ�ֱ�����ϼ��ͼ���Ե��
# ������ img_sobel_x ��ͬ��ֻ�� x �� y ����ĵ������������ˡ�
img_sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

# ʹ�� Laplacian ���ӽ���ͼ���Ե��⡣
# cv2.Laplacian �������ڼ���ͼ��Ķ��׵�����
# img_gray: ����ͼ�񣬱����ǵ�ͨ��ͼ��
# cv2.CV_64F: ���ͼ����������ͣ�����ѡ�� 64 λ�������Ա��������
# ksize: Laplacian ���ӵľ���˴�С��3 ��ʾ 3x3 �ľ���ˡ�
img_laplace = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=3)

# ʹ�� Canny ���ӽ���ͼ���Ե��⡣
# cv2.Canny �������ڼ��ͼ���еı�Ե��
# img_gray: ����ͼ�񣬱����ǵ�ͨ��ͼ��
# 100: ��Ե���ĵ���ֵ��
# 150: ��Ե���ĸ���ֵ��
# Canny ����ͨ��˫��ֵ����ȷ����Щ��Ե�������ı�Ե��
img_canny = cv2.Canny(img_gray, 100, 150)

# ����һ�� 2x3 ����ͼ���֣�����ʾԭʼ�Ҷ�ͼ��
# plt.subplot(231) ��ʾ�� 2x3 ����ͼ������ѡ��� 1 ��λ�á�
# plt.imshow(img_gray, "gray") ������ʾͼ��"gray" ��ʾʹ�ûҶ���ɫӳ�䡣
# plt.title("original") ����������ͼ�ı��⡣
plt.subplot(231), plt.imshow(img_gray, "gray"), plt.title("original")

# ����һ�� 2x3 ����ͼ���֣�����ʾ Sobel ������ˮƽ�����ϵı�Ե�������
# plt.subplot(232) ��ʾ�� 2x3 ����ͼ������ѡ��� 2 ��λ�á�
# plt.imshow(img_sobel_x, "gray") ������ʾͼ��"gray" ��ʾʹ�ûҶ���ɫӳ�䡣
# plt.title("sobel_x") ����������ͼ�ı��⡣
plt.subplot(232), plt.imshow(img_sobel_x, "gray"), plt.title("sobel_x")

# ����һ�� 2x3 ����ͼ���֣�����ʾ Sobel �����ڴ�ֱ�����ϵı�Ե�������
# plt.subplot(233) ��ʾ�� 2x3 ����ͼ������ѡ��� 3 ��λ�á�
# plt.imshow(img_sobel_y, "gray") ������ʾͼ��"gray" ��ʾʹ�ûҶ���ɫӳ�䡣
# plt.title("sobel_y") ����������ͼ�ı��⡣
plt.subplot(233), plt.imshow(img_sobel_y, "gray"), plt.title("sobel_y")

# ����һ�� 2x3 ����ͼ���֣�����ʾ Laplacian ���ӵı�Ե�������
# plt.subplot(234) ��ʾ�� 2x3 ����ͼ������ѡ��� 4 ��λ�á�
# plt.imshow(img_laplace, "gray") ������ʾͼ��"gray" ��ʾʹ�ûҶ���ɫӳ�䡣
# plt.title("laplace") ����������ͼ�ı��⡣
plt.subplot(234), plt.imshow(img_laplace, "gray"), plt.title("laplace")

# ����һ�� 2x3 ����ͼ���֣�����ʾ Canny ���ӵı�Ե�������
# plt.subplot(235) ��ʾ�� 2x3 ����ͼ