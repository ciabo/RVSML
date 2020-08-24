import numpy as np
import dtw
import OPW
from scipy.spatial.distance import cdist

def optimize(trainset, templateNum, l, err_limit, align_algorithm="dtw", lambda1=50, lambda2=0.1, sigma=1,
             previousExercise=None, exercise=None):
    """
    :param trainset: list containing the training sequences divided by class
                        trainset: a list with size of (1, the number of classes);
                        for the c-th class, trainset[c]: a list with size of (1, the number of training sequences for the c-th class)
                        for the i-th sequence sample, trainset[c][i]: a numpy matrix with size of (the length of the sequence, the dimensionality of vectors in this sequence)
    :param templateNum: dimensions of virtual sequences
    :param l: lambda value
    :param err_limit: bottom limit for the error to stop the optimization
    :param align_algorithm: algorithm used to align sequences "dtw" or "opw"
    :param lambda1: lambda 1 value used in OPW
    :param lambda2: lambda 2 value used in OPW
    :param sigma: sigma value used in OPW
    :return: the value of L in the decomposizìtion of M in mahalanobis distance
    """
    # create virtual sequences V
    classNum = len(trainset)
    trainsetnum = np.zeros(shape=(classNum))
    V = []
    activedim = 0
    downdim = classNum * templateNum
    dim = trainset[0][0].shape[1]
    for c in range(classNum):
        trainsetnum[c] = len(trainset[c])
        V.append(np.zeros(shape=(templateNum, downdim)))
        for a in range(templateNum):
            V[c][a][activedim] = 1
            activedim += 1
    # initialize the alignment matrices T
    N = np.sum(trainsetnum)

    L_a = np.zeros(shape=(dim, dim))
    L_b = np.zeros(shape=(dim, downdim))
    for c in range(classNum):
        for n in range(len(trainset[c])):
            seqLen = trainset[c][n].shape[0]
            T_ini = np.ones(shape=(seqLen,templateNum)) / (seqLen*templateNum)
            for i in range(seqLen):
                k = np.expand_dims(trainset[c][n][i], 0)
                temp_ra = np.transpose(k).dot(k)
                for j in range(templateNum):
                    L_a = L_a + T_ini[i][j] * temp_ra
                    L_b = L_b + T_ini[i][j] * np.transpose(k).dot(np.expand_dims(V[c][j], 0))
    L_i = L_a + l * N * np.identity(dim)
    L = np.linalg.solve(L_i, L_b)


    # optimization
    loss_old = 10 ^ 8
    maxIterations = 1000
    for k in range(maxIterations):
        printProgressBar(k, maxIterations, 'Iterations of '  + ': ' + str(k) + '/' + str(maxIterations))
        loss = 0
        L_a = np.zeros(shape=(dim, dim))
        L_b = np.zeros(shape=(dim, downdim))
        for c in range(classNum):
            for n in range(len(trainset[c])):
                seqLen = trainset[c][n].shape[0]
                if align_algorithm == 'dtw':
                    d, T = dtw2(np.transpose(trainset[c][n].dot(L)), np.transpose(V[c]))
                else:
                    d, T = opw(trainset[c][n].dot(L), V[c], a=None, b=None, lambda1=lambda1, lambda2=lambda2, sigma=sigma, VERBOSE=0)
                loss = loss + d
                for i in range(seqLen):
                    k = np.expand_dims(trainset[c][n][i], 0)
                    temp_ra = np.transpose(k).dot(k)
                    for j in range(templateNum):
                        L_a = L_a + T[i][j] * temp_ra
                        L_b = L_b + T[i][j] * np.transpose(k).dot(np.expand_dims(V[c][j], 0))

        loss = loss / N + np.trace(np.transpose(L).dot(L))
        if abs(loss - loss_old) < err_limit:
            break
        else:
            loss_old = loss
        L_i = L_a + l * N * np.identity(dim)
        L = np.linalg.solve(L_i, L_b)
    return L
