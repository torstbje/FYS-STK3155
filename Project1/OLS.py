import numpy as np
import matplotlib.pyplot as plt
import random as random
from CustomFunctions import*


n = input("Choose n: ")

x = np.random.rand(n,1)
y = np.random.rand(n,1)
z_d = np.zeros(n)
for i in range(n):
    z_d[i] = FrankeFunction(x[i],y[i]) + 0.01*np.random.normal(0,1)
dim_list = np.linspace(0,5,6)
MSE_list = np.zeros(len(dim_list))
for j in range(len(dim_list)):

    dim = int(dim_list[j])
    X = designMatrix(x,y,dim)

    beta = findBeta(X,z_d)
    z = np.dot(X,beta)
    MSE_list[j] = MSE(z_d,z)
    #print(z,z_d)
print MSE_list
