# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext_format_version: '1.2'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.6.5
# ---

# +
'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)

Output:     the most popular class label

@author: pbharrin
'''
import numpy as np #引入numpy模块
import operator #引入运算符模块
from os import listdir #os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
#这个列表以字母顺序。 它不包括 '.' 和'..' 即使它在文件夹中。

#只支持在 Unix, Windows 下使用。
# -

# ####K近邻算法

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]         #数据集大小
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet #
    sqDiffMat = diffMat**2                             #   距离计算
    sqDistances = sqDiffMat.sum(axis=1)                #
    distances = sqDistances **0.5       #
    sortedDistIndicies = distances.argsort()          # 返回距离值从小到大排列的索引值
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]    #获取标签
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  #出现就+1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) #按照第二个元素的次序排序，频数排序，逆序
    return sortedClassCount[0][0]  #返回频数最高的标签

def createDataSet():     #创建数据集和标签
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])   #数据集
    labels = ['A', 'A', 'B', 'B']          #标签
    return group, labels

group, labels = createDataSet()
print (group)
print (labels)

def file2matrix(filename):
    love_dictionary = {'largeDoses':3, 'smallDoses':2, 'didntLike':1}   #定义三种类型的标签
    fr = open(filename)  #打开文件
    arrayOLines = fr.readlines() #读取行
    numberOfLines = len(arrayOLines)            #得到文件行数
    returnMat = np.zeros((numberOfLines, 3))        #创建返回的numpy矩阵
    classLabelVector = []                       #prepare labels return
    index = 0
    #创建数据集
    for line in arrayOLines:  #读取每一行
        line = line.strip()  #拆分行
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]  #数据集
        #创建标签数据
        if(listFromLine[-1].isdigit()):  #每一行的最后一列为标签，查看是否只含有数字
            classLabelVector.append(int(listFromLine[-1]))  #添加标签
        else:
            classLabelVector.append(love_dictionary.get(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


datingDataMat,datingLabels = file2matrix(r'C:\Users\Administrator\Documents\GitHub\MachineLearningInAction-Camp\Week1\Reference Code\datingTestSet2.txt')
print(datingDataMat)
print(datingLabels)

import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*np.array(datingLabels),15.0*np.array(datingLabels))
plt.show()

from matplotlib.font_manager import FontProperties 
font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
#ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*np.array(datingLabels),15.0*np.array(datingLabels)) 
datingLabels = np.array(datingLabels)
idx_1 = np.where(datingLabels==1) 
p1 = ax.scatter(datingDataMat[idx_1,0],datingDataMat[idx_1,1],marker = '*',color = 'r',label='1',s=10) 
idx_2 = np.where(datingLabels==2) 
p2 = ax.scatter(datingDataMat[idx_2,0],datingDataMat[idx_2,1],marker = 'o',color ='g',label='2',s=20)
idx_3 = np.where(datingLabels==3)
p3 = ax.scatter(datingDataMat[idx_3,0],datingDataMat[idx_3,1],marker = '+',color ='b',label='3',s=30)
plt.xlabel("每年获取的飞行里程数",fontproperties=font)
plt.ylabel("玩视频游戏所消耗的事件百分比",fontproperties=font)
ax.legend((p1,p2,p3),("不喜欢","魅力一般","极具魅力"),loc=2,prop=font)
plt.show()

# #### newValue = (oldVlaue-min)/(max -min)

###归一化###
def autoNorm(dataSet):
    minVals = dataSet.min(0)   #从列中选取最小值
    maxVals = dataSet.max(0)   #从列中选取最大值
    ranges = maxVals - minVals #
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet/np.tile(ranges, (m, 1))   #element wise divide
    return normDataSet, ranges, minVals

normMat, ranges, minVals = autoNorm(datingDataMat)
print(normMat, ranges,minVals)

def datingClassTest():
    hoRatio = 0.50      #测试集比例为10%
    datingDataMat, datingLabels = file2matrix(r'C:\Users\Administrator\Documents\GitHub\MachineLearningInAction-Camp\Week1\Reference Code\datingTestSet.txt')       #加载数据集
    normMat, ranges, minVals = autoNorm(datingDataMat)                    #归一化
    m = normMat.shape[0]                                                  #数据集大小
    numTestVecs = int(m*hoRatio)                                          #
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)#测试集，训练集，标签，k
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0   #错误的
    print("the total error rate is: %f" % (errorCount / float(numTestVecs))) #错误率
   # print(errorCount)

datingClassTest()

def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']   #标签
    percentTats = float(input(\
                                  "percentage of time spent playing video games?"))  #输入玩视频游戏所耗时百分比
    ffMiles = float(input("frequent flier miles earned per year?"))                  #输入每年获得的飞行常客里程数
    iceCream = float(input("liters of ice cream consumed per year?"))                #输入每周消费的冰淇淋公升数
    datingDataMat, datingLabels = file2matrix(r'C:\Users\Administrator\Documents\GitHub\MachineLearningInAction-Camp\Week1\Reference Code\datingTestSet2.txt')                  #读取数据集
    normMat, ranges, minVals = autoNorm(datingDataMat)                              #归一化
    inArr = np.array([ffMiles, percentTats, iceCream,])                             #向量化
    classifierResult = classify0((inArr - \
                                  minVals)/ranges, normMat, datingLabels, 3)        #进入算法
    print("You will probably like this person: %s" % resultList[classifierResult - 1]) #输出类型

classifyPerson()

def img2vector(filename):
    returnVect = np.zeros((1, 1024))  #创建一个1*1024的向量
    fr = open(filename)               #打开文件
    for i in range(32):               #二进制矩阵的每一行
        lineStr = fr.readline()       #读取每一行
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

testVector = img2vector(r'C:\Users\Administrator\Documents\GitHub\MachineLearningInAction-Camp\Week1\Reference Code\digits\testDigits\0_13.txt')
testVector[0,32:63]

def handwritingClassTest():
    hwLabels = []  #建立存放数字标签的列表
    trainingFileList = listdir('trainingDigits')           #返回指定的文件夹包含的文件或文件夹的名字的列表
    m = len(trainingFileList) #文件数目
    trainingMat = np.zeros((m, 1024))   #
    for i in range(m):
        fileNameStr = trainingFileList[i]   #读取每一个文件夹
        fileStr = fileNameStr.split('.')[0]     #拆分文件夹中的每一个文件名
        classNumStr = int(fileStr.split('_')[0])    #获取数字标签
        hwLabels.append(classNumStr)    
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)  #读取训练集
    testFileList = listdir('testDigits')        #测试数据
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))

handwritingClassTest()


