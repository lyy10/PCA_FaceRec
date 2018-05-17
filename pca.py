# PCA

import math
import numpy
import matplotlib.pyplot as v_image
import matplotlib.image as r_image

############### 定义测试：训练比例参数 比例相加为10 贡献率P ##
i_test = 5
i_train = 5
P = 0.9
############### 读入人脸库 ######################
train = numpy.empty((1,10304))
xmean = numpy.empty((i_train*40,10304))
for d in range(1,41):
    for i in range(1,i_train+1):
        image = r_image.imread('./ORL/s' + str(d) + '/' + str(i) + '.jpg')
#v_image.imshow(image,cmap='Greys_r')
#v_image.show()
        image = image.T.flatten()
        train = numpy.vstack((train,image))
train = train[1:,:]
# 平均人脸
average = train.mean(axis=0)
for i in range(0,i_train*40):
    xmean[i] = train[i] - average
# 计算 K-L 变换的生成矩阵
m = xmean.dot(xmean.T)
# 求特征值 特征向量
d,v = numpy.linalg.eig(m)
evals = numpy.argsort(d)
d = numpy.sort(-d)
d = -d#特征值
# 降维
dsum = d.sum()
dsum_extract = 0
p = -1
while dsum_extract/dsum < P: # 贡献率值
    p += 1
    dsum_extract = d[0:p].sum()
print("贡献率: "+str(p))
vv = v[:,evals[-p:]]
vv = numpy.fliplr(vv)
######### 开始训练 ############
base = numpy.dot(numpy.dot(xmean.T, vv), numpy.diag(1/numpy.sqrt(d[0:p])))
allcoor = train.dot(base)
######## 开始测试 ############
accu = 0
mdist = numpy.empty((i_train*40))
for i in range(1,41):
    for d in range(i_train+1,11):
        image = r_image.imread('./ORL/s' + str(i) + '/' + str(d) + '.jpg')
        image = image.T.flatten()
        tcoor = image.dot(base)
        for k in range(0,i_train*40):
            mdist[k] = numpy.linalg.norm(tcoor - allcoor[k])
        index = numpy.argsort(mdist)
        ###################三阶####################
        #print("三阶近邻:")
        #class1 = math.floor((index[0])/i_train) + 1
        #class2 = math.floor((index[1])/i_train) + 1
        #class3 = math.floor((index[2])/i_train) + 1
        #if class1 != class2 & class2 != class3:
        #    classi = class1
        #elif class1 == class2:
        #    classi = class1
        #elif class2 == class3:
        #    classi = class2
        ##################四阶#####################
        print("四阶近邻:")
        classs = numpy.array([0,0,0,0])
        classs[0] = math.floor((index[0])/i_train) + 1
        classs[1] = math.floor((index[1])/i_train) + 1
        classs[2] = math.floor((index[2])/i_train) + 1
        classs[3] = math.floor((index[3])/i_train) + 1
        num_class = -1
        for k in range(0,4):
            if num_class < len(classs[classs==classs[k]]):
                num_class = len(classs[classs==classs[k]])
                classi = classs[k]
        ###############################################
        if classi == i:
            accu = accu + 1
print(accu/(i_test*40))
