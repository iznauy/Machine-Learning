import numpy as np

def ridgeRegression(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(xTx) == 0.0:
        print 'The matrix is singular, cannot do inverse'
        return
    ws = denom.I * xMat.T * yMat
    return ws
