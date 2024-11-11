#!/usr/bin/env python
# encoding=gbk
import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient ]]])   
��Ҫ������
��һ����������Ҫ�����ԭͼ�񣬸�ͼ�����Ϊ��ͨ���ĻҶ�ͼ��
�ڶ�����������ֵ1��
��������������ֵ2��
'''

img = cv2.imread("../week02/lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("canny", cv2.Canny(gray, 200, 300))

'''
�ֶ�ʵ��
1.ͼ�λҶȻ�
2.��˹ƽ��
3.��Ե��� 
4.�Ǽ���ֵ����
5.˫��ֵ�㷨�������ӱ�Ե
'''
######  1 .ͼ�λҶȻ�  #################
img = cv2.imread("../week02/lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

'''
�����˹�˺���
g(x,y)=1/2*PI*sigma^2 *(e^-(x^2+y^2)/(2*sigma^2))
'''
sigma = 0.5
p1 = 1 / (2 * np.pi * sigma ** 2)
p2 = -1 / 2 * sigma ** 2
dim = 5  # ��˹�˳ߴ�
gauss_filter = np.zeros([dim, dim])  # �洢��˹��
# ��X,Y ������ [-dim/2,dim/2] i ӳ��Ϊ tmp [i]�е�ֵ
tmp = np.zeros(dim)
for i in range(dim):
    tmp[i] = i - dim // 2

for i in range(dim):
    for j in range(dim):
        gauss_filter[i, j] = p1 * np.exp((tmp[i] ** 2 + tmp[j] ** 2) * p2)

# ��һ����˹��
# _range = np.max(gauss_filter) - np.min(gauss_filter)
# gauss_filter = (gauss_filter - np.min(gauss_filter)) / _range
gauss_filter = gauss_filter / gauss_filter.sum()

# ��ԭͼ�����
dx, dy = gray.shape
# �洢ƽ��֮���ͼ��zeros�����õ����Ǹ���������
img_new = np.zeros(gray.shape)
tmp = dim // 2
# ��Ե� �������
'''
��һ��Ԫ��(before_1, after_1)��ʾ��һά���С�����䷽ʽ��ǰ�����before_1����ֵ���������after_1����ֵ
��2��Ԫ��(before_2, after_2)��ʾ�ڶ�ά���С�����䷽ʽ��ǰ�����before_2����ֵ���������after_2����ֵ
a = np.array([1, 2, 3, 4, 5])
a=np.pad(a,(2,4),'constant')
[0 0 1 2 3 4 5 0 0 0 0]
      0 0 0 0 0 0 0
      0 0 0 0 0 0 0
      0 0 x x x 0 0
      0 0 x x x 0 0
      0 0 x x x 0 0
      0 0 0 0 0 0 0
      0 0 0 0 0 0 0  
'''
img_gauss_pad = np.pad(gray, ((tmp, tmp), (tmp, tmp)), 'constant')
for i in range(dx):
    for j in range(dy):
        img_new[i, j] = np.sum(img_gauss_pad[i:i + dim, j:j + dim]*gauss_filter)

plt.figure(1)
plt.imshow(img_new.astype(np.uint8), cmap='gray')  # ��ʱ��img_new��255�ĸ��������ݣ�ǿ������ת���ſ��ԣ�gray�ҽ�
plt.axis('off')

#######  2�����ݶȡ�sobel����  #############################
'''
�ݶ�f(x,y) =  df(x,y)/dx,df(x,y)/dy
'''
soble_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
soble_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
# �洢�ݶ�ͼ��
img_tidu_x = np.zeros(img_new.shape)  # �洢�ݶ�ͼ��
img_tidu_y = np.zeros([dx, dy])
# �ݶ�ֵ
img_tidu = np.zeros(img_new.shape)
'''
       0 0 0 0 0 
       0 x x x 0 
       0 x x x 0 
       0 x x x 0 
       0 0 0 0 0 
'''
img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')  # ��Ե��������������ṹ����д1
for i in range(dx):
    for j in range(dy):
        img_tidu_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3]*soble_x)
        img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3]*soble_y)
        img_tidu[i,j] = np.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)
# ��ֹ x =0 Ϊ�����
img_tidu_x[img_tidu_x == 0] = 0.00000001
# �ݶȽǶ�
angle = img_tidu_y / img_tidu_x

plt.figure(2)
plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
plt.axis('off')

#######  3���Ǽ���ֵ����  #############################
img_yizhi = np.zeros(img_tidu.shape)
for i in range(1, dx - 1):
    for j in range(1, dy - 1):
        flag = True  # ��8�������Ƿ�ҪĨȥ�������
        # y=  y = y1*(1-k) +y2*k => y= (y2-y1)k + y1
        temp = img_tidu[i - 1:i + 2, j - 1:j + 2]  # �ݶȷ�ֵ��8�������
        if angle[i, j] <= -1:  # ʹ�����Բ�ֵ���ж��������
            num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
            num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
            if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                flag = False
        elif angle[i, j] >= 1:
            num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
            num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
            if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                flag = False
        elif angle[i, j] > 0:
            num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
            num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
            if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                flag = False
        elif angle[i, j] < 0:
            num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
            num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
            if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                flag = False
        if flag:
            img_yizhi[i, j] = img_tidu[i, j]

plt.figure(3)
plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
plt.axis('off')

# 4��˫��ֵ��⣬���ӱ�Ե����������һ���Ǳߵĵ�,�鿴8�����Ƿ�����п����Ǳߵĵ㣬��ջ
lower_boundary =img_tidu.mean() * 0.5  #128
print(lower_boundary)
high_boundary = lower_boundary * 3  # ���������ø���ֵ�ǵ���ֵ������
zhan = []
for i in range(1, img_yizhi.shape[0] - 1):  # ��Ȧ��������
    for j in range(1, img_yizhi.shape[1] - 1):
        #���Ϊǿ��Ե
        if img_yizhi[i, j] >= high_boundary:  # ȡ��һ���Ǳߵĵ�
            img_yizhi[i, j] = 255
            zhan.append([i, j])
        #���ر�����
        elif img_yizhi[i, j] <= lower_boundary:  # ��
            img_yizhi[i, j] = 0

#����Ե����  ͨ���鿴����Ե���ؼ���8���������أ�ֻҪ����һ��Ϊǿ��Ե���أ��������Ե��Ϳ��Ա���Ϊ��ʵ�ı�Ե

while not len(zhan) == 0 :
    temp_1, temp_2 = zhan.pop()  # ��ջ
    a = img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
    if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
        img_yizhi[temp_1 - 1, temp_2 - 1] = 255  # ������ص���Ϊ��Ե
        zhan.append([temp_1 - 1, temp_2 - 1])  # ��ջ
    if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
        img_yizhi[temp_1 - 1, temp_2] = 255
        zhan.append([temp_1 - 1, temp_2])
    if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
        img_yizhi[temp_1 - 1, temp_2 + 1] = 255
        zhan.append([temp_1 - 1, temp_2 + 1])
    if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
        img_yizhi[temp_1, temp_2 - 1] = 255
        zhan.append([temp_1, temp_2 - 1])
    if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
        img_yizhi[temp_1, temp_2 + 1] = 255
        zhan.append([temp_1, temp_2 + 1])
    if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
        img_yizhi[temp_1 + 1, temp_2 - 1] = 255
        zhan.append([temp_1 + 1, temp_2 - 1])
    if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
        img_yizhi[temp_1 + 1, temp_2] = 255
        zhan.append([temp_1 + 1, temp_2])
    if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
        img_yizhi[temp_1 + 1, temp_2 + 1] = 255
        zhan.append([temp_1 + 1, temp_2 + 1])

#�ٴ�����
for i in range(img_yizhi.shape[0]):
    for j in range(img_yizhi.shape[1]):
        if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
            img_yizhi[i, j] = 0

#��ͼ
plt.figure(4)
plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
plt.axis('off')  # �ر�����̶�ֵ


plt.show()
