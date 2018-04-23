# PCA

import math
import numpy
import matplotlib.pyplot as v_image
import matplotlib.image as r_image

train = numpy.empty((1,10304))
xmean = numpy.empty((200,10304))
for d in range(1,41):
    for i in range(1,6):
        image = r_image.imread('./ORL/s' + str(d) + '/' + str(i) + '.jpg')
#v_image.imshow(image,cmap='Greys_r')
#v_image.show()
        image = image.T.flatten()
        train = numpy.vstack((train,image))
train = train[1:,:]
average = train.mean(axis=0)
for i in range(0,200):
    xmean[i] = train[i] - average
m = xmean.dot(xmean.T)


d,v = numpy.linalg.eig(m)
#print(v)
#print(d)
evals = numpy.argsort(d)
d = numpy.sort(-d)
d = -d#特征值
#v = numpy.fliplr(v)#特征向量
#print(v.shape)
dsum = d.sum()
dsum_extract = 0
p = -1
#print(d)
while dsum_extract/dsum < 0.9:
    p += 1
    dsum_extract = d[0:p].sum()
#print(p)
vv = v[:,evals[:-p-1:-1]]
#vv = numpy.fliplr(vv)
######### 开始训练 ############
#A = train.T
#print(A.shape)
#B = v[:,0:p]
#print(B.shape)
#C = numpy.diag(1/numpy.sqrt(d[0:p]))#numpy.diag(numpy.power(d[0:p],-float(1)/2))
#print(C)
base = numpy.dot(numpy.dot(train.T, vv), numpy.diag(1/numpy.sqrt(d[0:p])))
#print(base)
allcoor = train.dot(base)
#print(allcoor)

######## 开始测试 ############
accu = 0
mdist = numpy.empty((200))
for i in range(1,41):
    for d in range(6,11):
        image = r_image.imread('./ORL/s' + str(i) + '/' + str(d) + '.jpg')
        image = image.T.flatten()
        tcoor = image.dot(base)
        for k in range(0,200):
            mdist[k] = numpy.linalg.norm(tcoor - allcoor[k])
        index = numpy.argsort(mdist)
        #index = index.flatten()
        #print(index)
        class1 = math.floor((index[0]-1)/5) + 1
        class2 = math.floor((index[1]-1)/5) + 1
        class3 = math.floor((index[2]-1)/5) + 1
        if class1 != class2 & class2 != class3:
            classi = class1
        elif class1 == class2:
            classi = class1
        elif class2 == class3:
            classi = class2
        if classi == i:
            accu = accu + 1
print(accu/200)
