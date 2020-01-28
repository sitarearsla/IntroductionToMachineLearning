#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def sigmoid(A):
    return 1 / (1 + np.exp(-A))


def sigmoid_cost(pY, Y):
    return ((np.square(np.abs(pY - Y))).sum()) / 2


def get_data():
    df = pd.read_csv('hw02_images.csv', header=None)
    X = df.to_numpy()
    df2 = pd.read_csv('hw02_labels.csv', header=None)
    Y = df2.to_numpy()
    return X, Y


def get_weights():
    df3 = pd.read_csv('initial_W.csv', header=None)
    W = df3.to_numpy()
    df4 = pd.read_csv('initial_w0.csv', header=None)
    w0 = df4.to_numpy()
    w0 = w0.reshape(-1, )
    return W, w0


class Regression(object):
    def __init__(self):
        self.W = None
        self.w0 = None

    def fit(self, X, y_formatted, eta=0.0001, epsilon=1e-3, max_iteration=500, show_plot=False, ):
        self.W, self.w0 = get_weights()
        errors = []

        # calculate the prob
        for i in range(max_iteration):
            pY = self.forward(X)
            self.w0 += eta * ((y_formatted - pY) * pY * (1 - pY)).sum(axis=0)
            self.W += eta * (X.T.dot((y_formatted - pY) * pY * (1 - pY)))
            ##for the error graph
            err = sigmoid_cost(pY, y_formatted)
            errors.append(err)

        if show_plot:
            plt.plot(errors)
            plt.show()

    def forward(self, X):
        return sigmoid(X.dot(self.W) + self.w0)

    def predict(self, X):  
        return self.forward(X).argmax(axis=1)


if __name__ == '__main__':
    X, Y = get_data()
    Y = Y.reshape(-1, )
    Ntrain = int(np.shape(Y)[0] / 2.0)
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    y_formatted = np.zeros((500, 5))
    for i in range(Xtrain.shape[0]):
        y_formatted[i, Ytrain[i] - 1] = 1

    model = Regression()
    model.fit(Xtrain, y_formatted, show_plot=True)

    trainResults = confusion_matrix(Ytrain, model.predict(Xtrain))
    print(trainResults)

    testResults = confusion_matrix(Ytest, model.predict(Xtest))
    print(testResults)


# In[ ]:




