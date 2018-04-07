import numpy as np
from numpy.linalg import solve
import findMin
from scipy.optimize import approx_fprime
import utils

class logReg:
    # Logistic Regression
    def __init__(self, verbose=2, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self,X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)
    def predict(self, X):
        return np.sign(X@self.w)

class logRegL2():
    # Logistic Regression
    def __init__(self, verbose=2, maxEvals=400, lammy=1.0):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.lammy = lammy
        self.bias = True

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw))) + self.lammy/2*(w.T.dot(w))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res) + self.lammy*w

        return f, g

    def fit(self,X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)
    def predict(self, X):
        return np.sign(X@self.w)
    
class logRegL1():
    # Logistic Regression with L1-regularization
    def __init__(self, verbose=2, maxEvals=400, L1_lambda=1.0):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.L1_lambda = L1_lambda
        self.bias = True

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw))) 

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res) 

        return f, g

    def fit(self,X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMinL1(self.funObj, self.w, self.L1_lambda,
                                      self.maxEvals, X, y, verbose=self.verbose)
    def predict(self, X):
        return np.sign(X@self.w)

class logRegL0(logReg):
    # L0 Regularized Logistic Regression
    def __init__(self, L0_lambda=1.0, verbose=2, maxEvals=400):
        self.verbose = verbose
        self.L0_lambda = L0_lambda
        self.maxEvals = maxEvals


    def fit(self, X, y):
        n, d = X.shape
        minimize = lambda ind: findMin.findMin(self.funObj,
                                                  np.zeros(len(ind)),
                                                  self.maxEvals,
                                                  X[:, ind], y, verbose=0)
        selected = set()
        selected.add(0)
        minLoss = np.inf
        oldLoss = 0
        bestFeature = -1

        while minLoss != oldLoss:  # loop untiil Etrain = 0
            oldLoss = minLoss
            print("Epoch %d " % len(selected))
            print("Selected feature: %d" % (bestFeature))
            print("Min Loss: %.3f\n" % minLoss)

            for i in range(d):
                if i in selected:
                    continue

                selected_new = selected | {i}
                # TODO for Q2.3: Fit the model with 'i' added to the features,
                # then compute the loss and update the minLoss/bestFeature
                
                # use subset selected_new to fit model
                w_new,f = minimize(list(selected_new))
                X_new = X[:,list(selected_new)]
            
                
                # compute the score with these features
                f, g = self.funObj(w_new, X_new, y)
                L0penalty = (w_new != 0).sum() # LOnorm = number of non-zeros
                f = f + L0penalty
                
                # update the minloss/bestFeature
                if (f < minLoss):
                    minLoss = f
                    bestFeature = i
                

            selected.add(bestFeature)

        self.w = np.zeros(d)
        self.w[list(selected)], _ = minimize(list(selected))


class leastSquaresClassifier:
    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes,d))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1

            # solve the normal equations
            # with a bit of regularization for numerical reasons
            self.W[i] = np.linalg.solve(X.T@X+0.0001*np.eye(d), X.T@ytmp)

    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)

#  (maxEvals=500, verbose=0)
class logLinearClassifier(logReg):
    def __init__(self, maxEvals, verbose):
        self.maxEvals = maxEvals
        self.verbose = verbose
        pass
    
    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size
        
        # Initial guess
        self.W = np.zeros((d, self.n_classes))
        # in order to use the funObj function, we need to match the "yXw"
        # which is y(n*1) and Xw(n*d, d*k)
        # in theoretical analysis, W is k*d

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            # binary classifier
            ytmp[y==i] = 1
            ytmp[y!=i] = -1
            
            # logistic loss
            # funObj in logReg
            # findMin in findMin with gradient descent
            
            #utils.check_gradient(self, X, ytmp)
            (self.W[:,i], f) = findMin.findMin(self.funObj, self.W[:,i],
                self.maxEvals, X, ytmp, verbose = self.verbose)
            

        
    def predict(self, X):
        return np.argmax(X@self.W, axis = 1)
    

class softmaxClassifier():
    def __init__(self, maxEvals):
        self.maxEvals = maxEvals
        pass
    
    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size
        
        # initial guess
        self.W = np.zeros( (d, self.n_classes) )
        W_vector = self.W.flatten()
        (self.W, f) = findMin.findMin(self.softmaxFun, W_vector,
                self.maxEvals, X, y, verbose = 0)
        
        self.W = np.reshape(self.W, (d, self.n_classes))
    
    def predict(self, X):
        return np.argmax(X@self.W, axis = 1)
    
    def softmaxFun(self, W, X, y):
        n, d = X.shape
             
        k = self.n_classes
        W = np.reshape(W,(d,k))
        
        # compute the softmax loss function
        
        XW = X.dot(W)
        w_yixi = np.zeros((n))
        w_cxi = np.zeros((n,k))
        losseach = np.zeros((n))
        sum_w_cxi = np.zeros((n))
        for i in range(n):
            w_yixi[i] = XW[i,y[i]]
            for c in range(k):
                w_cxi[i,c] = np.exp(XW[i,c])
                #sum_w_cxi[i] = sum_w_cxi[i] + w_cxi[i,c]
            sum_w_cxi[i] = np.sum( w_cxi[i,:] )
            
            losseach[i] = -w_yixi[i]+np.log(sum_w_cxi[i])
        
        
        f = np.sum(losseach)
        
        # compute the gradient of softmax fun
        gradTemp = np.zeros((n,k))
        Indicator = np.zeros((n))
        prob = np.zeros( (n,k) )
        for c in range(k):
            for i in range(n):
                if c == y[i]:
                    Indicator[i] = 1
                else:
                    Indicator[i] = 0
                
            prob[:,c] = np.exp(XW[:,c])/sum_w_cxi
           
            gradTemp[:,c] = prob[:,c]-Indicator
        
        grad = X.T.dot(gradTemp)
        
        return f, grad.flatten()
        
        

        
  