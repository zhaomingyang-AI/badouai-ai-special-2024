import cv2
import numpy as np
import matplotlib.pyplot as plt

#读取原始图像
img = cv2.imread('lenna.png') 
print (img.shape)

#图像二维像素转换为一维
 # 将这个三维数组重新整形为一个二维数组，其中第一维的大小是自动计算的（由 -1 指定,即w*h），第二维的大小是 3。
 # data = img.reshape((-1,3))和data = img.reshape((rows * cols, 3))是相同的
data = img.reshape((-1,3)) 
data = np.float32(data)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)#停止条件(type,max_iter,epsilon)
flags = cv2.KMEANS_RANDOM_CENTERS#设置标签

compactness, labels2, centers2 = cv2.kmeans(data, 2, None, criteria, 10, flags)#K-Means聚类 聚集成2类
compactness, labels4, centers4 = cv2.kmeans(data, 4, None, criteria, 10, flags)#K-Means聚类 聚集成4类
compactness, labels8, centers8 = cv2.kmeans(data, 8, None, criteria, 10, flags)#K-Means聚类 聚集成8类
compactness, labels16, centers16 = cv2.kmeans(data, 16, None, criteria, 10, flags)#K-Means聚类 聚集成16类
compactness, labels64, centers64 = cv2.kmeans(data, 64, None, criteria, 10, flags)#K-Means聚类 聚集成64类

#图像转换回uint8二维类型
 # 质心转换为无符号 8 位整数类型，以便后续处理
 # 将 labels2 展平成一个一维数组，其长度为 512 * 512 = 262144。这个一维数组包含了所有像素的簇标签。
centers2 = np.uint8(centers2)
res = centers2[labels2.flatten()]
dst2 = res.reshape((img.shape))

centers4 = np.uint8(centers4)
res = centers4[labels4.flatten()]
dst4 = res.reshape((img.shape))

centers8 = np.uint8(centers8)
res = centers8[labels8.flatten()]
dst8 = res.reshape((img.shape))

centers16 = np.uint8(centers16)
res = centers16[labels16.flatten()]
dst16 = res.reshape((img.shape))

centers64 = np.uint8(centers64)
res = centers64[labels64.flatten()]
dst64 = res.reshape((img.shape))

#图像转换为RGB显示
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB)
dst4 = cv2.cvtColor(dst4, cv2.COLOR_BGR2RGB)
dst8 = cv2.cvtColor(dst8, cv2.COLOR_BGR2RGB)
dst16 = cv2.cvtColor(dst16, cv2.COLOR_BGR2RGB)
dst64 = cv2.cvtColor(dst64, cv2.COLOR_BGR2RGB)

#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

#显示图像
titles = [u'原始图像', u'聚类图像 K=2', u'聚类图像 K=4', u'聚类图像 K=8', u'聚类图像 K=16', u'聚类图像 K=64']
images = [img, dst2, dst4, dst8, dst16, dst64] 
for i in range(6):  
   plt.subplot(2,3,i+1), plt.imshow(images[i], 'gray'), 
   plt.title(titles[i])  
   plt.xticks([]),plt.yticks([])  
plt.show()
