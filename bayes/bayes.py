import numpy as np

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec


def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print 'The word %s is not in my dictionary' % word
    return returnVec


def bagOfWords2VecMN(vocabList, inputSet): # bag-of-words model
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def trainNB0(trainMatrix, trainCategory): # in fact, it's used in event models for text classfication
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass)
    p0 = sum(vec2Classify * p0Vec) + np.log(1 - pClass)
    if p1 > p0:
        return 1
    else:
        return 0


def anotherTrainNB(trainMatrix, trainCategory): # this comes from cs229, Andrew Ng
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p1Num = np.zeros(numWords)
    p0Num = np.zeros(numWords)
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
        else:
            p0Num += trainMatrix[i]
    p1Vect = (p1Num + 1) / float(sum(trainCategory) + 2) # Laplace smoothing
    p0Vect = (p0Num + 1) / float(numTrainDocs - sum(trainCategory) + 2)
    p1Vect = np.log(p1Vect) # avoid underflow
    p0Vect = np.log(p0Vect)
    return p0Vect, p1Vect, pAbusive


# testing spam mails
def textParse(bigString):
    import re
    listofTokens = re.split('\\W+', bigString)
    return [tokens.lower() for tokens in listofTokens if len(tokens) > 2]


def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('/Users/iznauy/Desktop/data/Ch04/email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('/Users/iznauy/Desktop/data/Ch04/email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainSet = range(50)
    testSet = []
    for i in range(10):
        randIndex = int(np.random.uniform(0, len(trainSet)))
        testSet.append(trainSet[randIndex])
        del(trainSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0.0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print 'Classification error', docList[docIndex]
    print 'The error rate is ', errorCount / len(testSet)