import numpy as np
from utils import euclidean_dist_squared
from numpy import linalg as LA

class Kmedians:

    def __init__(self, k):
        self.k = k

    def fit(self, X):
        N, D = X.shape
        y = np.ones(N)

        means = np.zeros((self.k, D))

        for kk in range(self.k):
            i = np.random.randint(N)
            means[kk] = X[i]
        

        while True:
            y_old = y
            dist1=np.zeros((N,self.k))
            for n in range(N):
                for kk in range(self.k):
                    diff = X[n, :] - means[kk, :]
                    dist1[n, kk] = LA.norm(diff, 1)  # Calculate Norm-1 distance
            # Compute euclidean distance to each mean
            dist1[np.isnan(dist1)] = np.inf
            y = np.argmin(dist1, axis=1)
            

            # Update means
            for kk in range(self.k):
                means[kk] = np.median(X[y == kk], axis=0)

            changes = np.sum(y != y_old)
            # print('Running K-means, changes in cluster assignment = {}'.format(changes))

            # Stop if no point changed cluster
            if changes == 0:
                break

        self.means = means
        

    def predict(self, X):
        means = self.means
        N, D = X.shape
        M, D = means.shape

        dist1 = np.zeros((N, M))
        for n in range(N):
            for m in range(M):
                diff = X[n, :] - means[m, :]
                dist1[n, m] = LA.norm(diff, 1)

        return np.argmin(dist1, axis=1)
       

    def error(self,X):
        means = self.means
        N, D = X.shape
        M, D = means.shape

        dist1 = np.zeros((N, M))
        for n in range(N):
            for m in range(M):
                diff = X[n, :] - means[m, :]
                dist1[n, m] = LA.norm(diff, 1)

        sortIndex = np.argmin(dist1, axis=1)
        minDist1 = np.zeros(N)
        for n in range(N):
            index = sortIndex[n]
            minDist1[n] = dist1[n, index]
        return np.sum(minDist1)
