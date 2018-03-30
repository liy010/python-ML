from numpy import *
import operator
from os import listdir

def createDataSet():
    group = array([[1.0, 1.1],[1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0] #dataSet数据0维的维数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet #tile函数构造数组
    sqDiffMat = diffMat**2 #每个元素的平方
    sqDistances = sqDiffMat.sum(axis=1) #列方向相加 array([2.21, 2.  , 0.  , 0.01])
    distances = sqDistances**0.5 #开方 
    sortedDistIndicies = distances.argsort() #排序,按当前排序后的序号生成的列表[2, 3, 1, 0]
    classCount={}
    for i in range(k):# [0, 1, 2]
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+ 1 #classCount{"B": 2, "A": 1}
    sortedClassCount = sorted(classCount.items(),#迭代 [("B", 2), ("A", 1)] 可以用for循环来提取键，值
                        key = operator.itemgetter(1), reverse=True) #operator.itemgetter(1):使用2,1来做为排序依据
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename) #
    arrayOLines = fr.readlines() #读取所有行
    numberOfLines = len(arrayOLines) #行的长度
    returnMat = zeros((numberOfLines, 3)) #创建一个0数组
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip() #用于移除字符串头尾指定的字符（默认为空格）
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        if listFromLine[-1] == "largeDoses": #把最后一列的喜欢，魅力，用数值表示
            classLabelVector.append(3)
        elif listFromLine[-1] == "smallDoses":
            classLabelVector.append(2)
        else:
            classLabelVector.append(1)
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                           datingLabels[numTestVecs:m], 3)
        print ("the classifier came back with: %d, the real answer is: %d"
                     %  (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): 
            errorCount += 1.0
    print ('the total error rate is: %f' % (errorCount/float(numTestVecs)))


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream condumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print ("You will probably like this person: ",resultList[classifierResult - 1])


def img2vector(filename):
    returnVect = zeros((1, 1024)) #创建一个1*1024数组
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()#读取一行
        for j in range(32):
            returnVect[0, 32*i + j] = int(lineStr[j])#平铺
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits') #列出文件夹下所有的文件
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i] 
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)   #
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr) #打开文件并处理
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest,
                                     trainingMat, hwLabels, 3)
        print ("the classifier came back with: %d, the real answer is: %d"
               % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print ("\nthe total number of errors is: %d" % errorCount)
    print ("\nthe total error rate is: %f" % (errorCount/float(mTest)))