from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed



def designMatrix(x,y,dim):
    '''
    Constructs the design matrix
    '''
    n = len(x)
    p = 1 + dim*(dim+3)/2       #Length of row vectors in matrix
    X = np.zeros((n,p))
    for i in range(n):
        '''
        Loops over number of data values
        '''
        k_start = 0
        k_end = 0
        for j in range(dim+1):
            '''
            Loops over polynomial degree
            '''
            k_start += j
            k_end += (j+1)
            for k in range(k_end-k_start):
                '''
                Loops over elements for a given polynomial degree
                '''
                p = k_start + k
                X[i,p] = x[i]**(j-k)*y[i]**k
    return X

def findBeta(X,y):
    '''
    Finds the optimal parameters beta for a given design matrix and data set
    '''
    Xt = X.transpose()
    X2 = np.dot(Xt,X)
    Xinv = np.linalg.inv(X2)
    Xf = np.dot(Xinv,Xt)
    return np.dot(Xf,y)


def MSE(y_real,y_sim):
    n = len(y_real)
    error = y_real - y_sim
    return np.dot(error,error)/n


def FrankeFunction(x,y):

    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4
