import numpy as np

def locallyWeightedLinearRegression(testPoint, xArr, yArr, k=0.5):
    xMat, yMat = np.mat(xArr), np.mat(yArr).T
    m = xMat.shape[0]
    weights = np.mat(np.eye(m))
    for j in range(m):
        diffMat = testPoint - xMat[j]
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * weights * xMat
    if np.linalg.det(xTx) == 0.0:
        print 'The matrix is singular, cannot do inverse'
        return
    ws = xTx.I * xMat.T * weights * yMat
    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=0.5):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = locallyWeightedLinearRegression(testArr[i], xArr, yArr, k)
    return yHat


def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()
