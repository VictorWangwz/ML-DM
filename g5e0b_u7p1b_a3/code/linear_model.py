import numpy as np
from numpy.linalg import solve
from findMin import findMin
from scipy.optimize import approx_fprime
import utils

# Ordinary Least Squares
class LeastSquares:
    def fit(self,X,y):
        self.w = solve(X.T@X, X.T@y)

    def predict(self, X):
        return X@self.w

# Least squares where each sample point X has a weight associated with it.
class WeightedLeastSquares(LeastSquares): # inherits the predict() function from LeastSquares
    def fit(self,X,y,z):
        ''' YOUR CODE HERE '''
        self.w = solve(X.T @ z @ X, X.T @ z @ y)

class LinearModelGradient(LeastSquares):

    def fit(self,X,y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros((d, 1))

        # check the gradient
        estimated_gradient = approx_fprime(self.w, lambda w: self.funObj(w,X,y)[0], epsilon=1e-6)
        implemented_gradient = self.funObj(self.w,X,y)[1]
        if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
            print('User and numerical derivatives differ: %s vs. %s' % (estimated_gradient, implemented_gradient));
        else:
            print('User and numerical derivatives agree.')

        self.w, f = findMin(self.funObj, self.w, 100, X, y)

    def funObj(self,w,X,y):

        ''' MODIFY THIS CODE '''
        # Calculate the function value
        # f = 0.5*np.sum((X@w - y)**2)

        f = np.sum(np.log(np.exp(X @ w - y) + np.exp(y - X @ w)))

        # Calculate the gradient value
        # g = X.T@(X@w-y)

        numerator = np.exp(X @ w - y) - np.exp(y - X @ w)
        denominator = np.exp(X @ w - y) + np.exp(y - X @ w)
        fraction = numerator / denominator

        g = X.T @ fraction

        return (f, g)


# Least Squares with a bias added
class LeastSquaresBias:

    def fit(self,X,y):
        ''' YOUR CODE HERE '''
        new_column = np.zeros(X.size)
        for x in range(len(new_column)):
            new_column[x] = 1
        X = np.concatenate((np.array(new_column)[:, np.newaxis], X), axis=1)

        self.w = solve(X.T @ X, X.T @ y)

    def predict(self, X):
        ''' YOUR CODE HERE '''
        new_column = np.zeros(X.size)
        for x in range(len(new_column)):
            new_column[x] = 1
        X = np.concatenate((np.array(new_column)[:, np.newaxis], X), axis=1)

        return X @ self.w

# Least Squares with polynomial basis
class LeastSquaresPoly:
    def __init__(self, p):
        self.leastSquares = LeastSquares()
        self.p = p

    def fit(self,X,y):
        ''' YOUR CODE HERE '''
        new_X = self.__polyBasis(X)
        self.w = solve(new_X.T @ new_X, new_X.T @ y)

    def predict(self, X):
        ''' YOUR CODE HERE '''
        new_X = self.__polyBasis(X)
        return new_X @ self.w

    # A private helper function to transform any matrix X into
    # the polynomial basis defined by this class at initialization
    # Returns the matrix Z that is the polynomial basis of X.
    def __polyBasis(self, X):
        ''' YOUR CODE HERE '''
        p = self.p

        new_new_X = np.tile(X, (1, p + 1))

        rows = X.size
        cols = p + 1

        for x in range(0, rows):
            for y in range(cols):
                new_new_X[x, y] = X[x] ** (y)

        return new_new_X

# Least Squares with RBF Kernel
class LeastSquaresRBF:
    def __init__(self, sigma):
        self.sigma = sigma

    def fit(self,X,y):
        self.X = X
        n, d = X.shape

        Z = self.__rbfBasis(X, X, self.sigma)

        # Solve least squares problem
        a = Z.T@Z + 1e-12*np.identity(n) # tiny bit of regularization
        b = Z.T@y
        self.w = solve(a,b)

    def predict(self, Xtest):
        Z = self.__rbfBasis(Xtest, self.X, self.sigma)
        yhat = Z@self.w
        return yhat

    def __rbfBasis(self, X1, X2, sigma):
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        d = X1.shape[1]
        den = 1 / np.sqrt(2 * np.pi * (sigma** 2))

        D = (X1**2)@np.ones((d, n2)) + \
            np.ones((n1, d))@(X2.T** 2) - \
            2 * (X1@X2.T)

        Z = den * np.exp(-1* D / (2 * (sigma**2)))
        return Z
