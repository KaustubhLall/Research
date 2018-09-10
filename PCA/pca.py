import numpy as np

def sanitizeInput(x, y):
    '''
    :params:
        x --> input of features. n*p matrix.
        y --> labels for each data point. n*1 matrix.

    :output:
        x, y --> sanitized for pca.
    '''
    pass

def shiftMean(x):
    '''
    Takes matrix x and shifts the mean such that each row has a sum of 0.
    '''
    for i in range(len(x)):
        mean = sum(x[i])/len(x[i])
        for j in range(len(x[i])):
            x[i][j] -= mean
    return x

def centerMatrix(x):
    '''
    Takes a matrix and centers it. Step is important (only) if high variance of features is an indicator of higher importance
    for the features.
    '''

def PCA(x, y, standardize=False):
    if standardize:
        z = centerMatrix(shiftMean(x))
    else:
        z = shiftMean(x)

    z = np.array(z)
    # covariance matrix = Z^T * Z
    cov = np.matmul(z.transpose(), z)
    print(np.array(cov))
    # fidn eignenvalues and eigenvectors as D and P respectively.
    D, P = np.linalg.eig(cov)
    # sort the eigenvalues and corresponding eigenvectors
    DP = [(D[i], P[i]) for i in range(len(D))]
    # sort the list by eigenvalue and reorder eigenvectors accordingly. Reverse = True sets descending order.
    DP = sorted(DP, key = lambda x : x[0], reverse = True)
    # extract P* (sorted eigenvectors)
    Pprime = [x[1] for x in DP]
    return np.matmul(z, Pprime)
    
x = [
        [7, 4, 3],
        [4, 1, 8],
        [6, 3, 5],
        [8, 6, 1],
        [8, 5, 7],
        [7, 2, 9],
        [5, 3, 3],
        [9, 5, 8],
        [7, 4, 5],
        [8, 2, 2]
        ]

y = [1, 2, 1]

print(PCA(x, y))
