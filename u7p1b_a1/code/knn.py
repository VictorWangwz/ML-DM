"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
from scipy import stats
import utils

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X # just memorize the trianing data
        self.y = y 

    def predict(self, Xtest):
        ''' YOUR CODE HERE FOR Q4.1.1 '''
        dict1=[]
        yhat=[]
        Xtest=np.array(Xtest)
        (t,n)=Xtest.shape
        dict1=utils.euclidean_dist_squared(self.X,Xtest)
        print(dict1.shape)
        for j in range(0,t):   
            temp=sorted(dict1[:,j])
            # temp=np.array(temp)
            # print(len(temp))
            # print(temp[0:self.k])
            temp1=np.array(temp[0:self.k])
            temp2=[np.where(dict1[:,j]==temp1[i]) for i in range(0,self.k)]
            temp3=np.array([self.y[i] for i in temp2])
            # print(temp3)
            yhat.append(utils.mode(temp3))
            # yhat.append(utils.mode(temp[0:self.k]))
        # print(yhat)
        # print(len(yhat))
        return(yhat)
        
        


        #raise NotImplementedError


class CNN(KNN):

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : an N by D numpy array
        y : an N by 1 numpy array of integers in {1,2,3,...,c}
        """

        Xcondensed = X[0:1,:]
        ycondensed = y[0:1]

        for i in range(1,len(X)):
            x_i = X[i:i+1,:]
            dist2 = utils.euclidean_dist_squared(Xcondensed, x_i)
            inds = np.argsort(dist2[:,0])
            yhat = utils.mode(ycondensed[inds[:min(self.k,len(Xcondensed))]])

            if yhat != y[i]:
                Xcondensed = np.append(Xcondensed, x_i, 0)
                ycondensed = np.append(ycondensed, y[i])

        self.X = Xcondensed
        self.y = ycondensed