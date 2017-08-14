import kNN
from os import listdir
import numpy as np

def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('/Users/iznauy/Desktop/data/Ch02/digits/trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m-1, 1024))
    for i in range(1, m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNum = int(fileStr.split('_')[0])
        hwLabels.append(classNum)
        trainingMat[i-1, :] = img2vector('/Users/iznauy/Desktop/data/Ch02/digits/trainingDigits/%s' % fileNameStr)
        testFileList = listdir('/Users/iznauy/Desktop/data/Ch02/digits/testDigits')
        error_count = 0.0
        mTest = len(testFileList)
    for i in range(1, mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNum = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('/Users/iznauy/Desktop/data/Ch02/digits/testDigits/%s' % fileNameStr)
        classifierResult = kNN.kNNclassifier(vectorUnderTest, trainingMat, hwLabels, 3)
        print 'The classifier came back with: %d, the real answer is %d' % (classifierResult, classNum)
        if classifierResult != classNum:
            error_count += 1
    print 'The total number of error: %d' % error_count
    print 'The total error rate is %f' % (error_count / (mTest - 1))
