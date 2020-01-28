#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def get_data():
    df = pd.read_csv('hw03_images.csv', header=None)
    X = df.to_numpy()
    df2 = pd.read_csv('hw03_labels.csv', header=None)
    Y = df2.to_numpy()
    return X, Y

def get_weights():
    df3 = pd.read_csv('initial_W.csv', header=None)
    W = df3.to_numpy()
    df4 = pd.read_csv('initial_V.csv', header=None)
    V = df4.to_numpy()
    return W, V

def derivative_V(Z, T, Y):
    return Z.T.dot(T - Y)

def derivative_W(X, Z, T, Y, V):
    dZ = (T - Y).dot(V.T) * Z * (1 - Z)
    return X.T.dot(dZ)

def derivative_W0(Z, T, Y, V):
    return ((T-Y).dot(V.T) * Z * (1 - Z)).sum(axis=0)

def derivative_V0(T, Y):
    return (T - Y).sum(axis=0)
    
def cost(T, Y):
    err = -(T * np.log(Y))
    return err.sum()

class Perceptron(object):
    def forward(self, X, W, V, W0, V0):
        Z = 1 / (1+ np.exp(-X.dot(W)-W0))
        A = Z.dot(V)+V0
        expA = np.exp(A)
        Y = expA / expA.sum(axis=1, keepdims=True)
        return Y, Z
    
    def fit(self, X, Y, W, V, W0, V0, eta = 0.0005, max_iteration = 500):
        costs = []
        
        K = len(set(Y.flatten())) #number of classes
        N = len(Y) 
    
        #indicator variable T
        T = np.zeros((N, K))
        for i in range(N):
            T[i, Y[i]-1] = 1
            
        for i in range(max_iteration):
            output, hidden = self.forward(X, W, V, W0, V0)
            c = cost(T, output)
            costs.append(c)
            
            #gradient ascent
            V += eta * derivative_V(hidden, T, output)
            W += eta * derivative_W(X, hidden, T, output, V)
            V0 += eta * derivative_V0(T, output)
            W0 += eta * derivative_W0(hidden, T, output, V)
        
        plt.plot(costs)
        plt.show()
        
    def predict(self, X, W, V, W0, V0):  
        return self.forward(X, W, V, W0, V0)[0].argmax(axis=1)
    
def main():
    X, Y = get_data()
    Y = Y.reshape(-1, )
    
    W, V = get_weights()
    W0 = W[:1]
    W = W[1:785]
    V0 = V[:1]
    V = V[1:21]
        
    Ntrain = int(np.shape(Y)[0] / 2.0)
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    model = Perceptron()
    model.fit(Xtrain, Ytrain, W, V, W0, V0)
    
    trainResults = confusion_matrix(Ytrain, model.predict(Xtrain, W, V, W0, V0))
    print(trainResults[1:6,:5])

    testResults = confusion_matrix(Ytest, model.predict(Xtest, W, V, W0, V0))
    print(testResults[1:6,:5])
    
if __name__ == '__main__':
    main()

