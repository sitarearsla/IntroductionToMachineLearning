#!/usr/bin/env python
# coding: utf-8

# In[46]:


import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal as multv
from sklearn.metrics import confusion_matrix

class NaiveBayes(object):   
    ##fit function takes in X, Y 
    def fit(self, X, Y):
        
        ##empy dictionary of gaussian parameters
        self.gaussians = dict()
        
        ##empty dictionary of priors
        self.priors = dict()
        
        ## get all the unique values in Y
        labels = set(Y.flatten())
        
        for c in labels: 
            ##Y equal to the current label c
            current_x = X[Y == c]
            self.gaussians[c] = {
                ##set the mean and variance
                'mean': current_x.mean(axis=0),
                'var': current_x.var(axis=0),
            }
            ##calculate prior
            self.priors[c] = float(np.shape(Y[Y == c])[0]) / np.shape(Y)[0]
       
        ##printing priors, mean and variance
        print(self.priors)
        print(self.gaussians)
    
    ##predict function takes in X
    def predict(self, X):
        N, D = X.shape
        
        ##number of classes
        K = len(self.gaussians)
        
        ##empty array of size N * K
        ##for each N samples, K different probabilities
        P = np.zeros((N, K))
        
        ##loop through all the gaussians
        for c, g in self.gaussians.items():
            mean, var = g['mean'], g['var']
            ##calculate the log of pdf
            P[:,c-1] = multv.logpdf(X, mean=mean, cov=var) + np.log(self.priors[c])
            
        ##returns an N sized array
        return np.argmax(P, axis=1)+1

if __name__ == '__main__':
        ##get data function
    def get_data():
        ##read X's from images.csv and put to numpy matrix
        df = pd.read_csv('hw01_images.csv', header=None)
        X = df.to_numpy()
        ##read Y's from labels.csv and put to numpy matrix
        df2 = pd.read_csv('hw01_labels.csv', header=None)
        Y = df2.to_numpy()
        return X, Y
    
    X, Y = get_data()
    ##reshape into column vector
    Y = Y.reshape(-1,)
    ##divide the datasets into 2 for training and testing
    Ntrain = int(np.shape(Y)[0]/2.0)
    ##set the train sets to the first half 
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    ##set the test set to the second half
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]
    
    ##create model
    model = NaiveBayes()
    
    ##fitting
    model.fit(Xtrain, Ytrain)
    
    trainResults = confusion_matrix(Ytrain, model.predict(Xtrain))
    print(trainResults)
    testResults = confusion_matrix(Ytest, model.predict(Xtest))
    print(testResults)


# In[ ]:




