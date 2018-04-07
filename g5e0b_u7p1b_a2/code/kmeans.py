import numpy as np
from utils import euclidean_dist_squared

class Kmeans:

    def __init__(self, k):
        self.k = k

    def fit(self, X):
        N, D = X.shape
        y = np.ones(N)

        means = np.zeros((self.k, D))
        for kk in range(self.k):
            i = np.random.randint(N)
            means[kk] = X[i]
        self.means=means

        while True:
            y_old = y

            # Compute euclidean distance to each mean
            dist2 = euclidean_dist_squared(X, means)
            dist2[np.isnan(dist2)] = np.inf
            y = np.argmin(dist2, axis=1)

            # Update means
            for kk in range(self.k):
                means[kk] = X[y==kk].mean(axis=0)

            changes = np.sum(y != y_old)
            # print('Running K-means, changes in cluster assignment = {}'.format(changes))
            # error=self.error(X)
            # print(error)

            # Stop if no point changed cluster
            if changes == 0:
                break

        self.means = means
        

    def predict(self, X):
        means = self.means
        dist2 = euclidean_dist_squared(X, means)
        dist2[np.isnan(dist2)] = np.inf
        return np.argmin(dist2, axis=1)

    def error(self,X):
        means = self.means
        N, D = X.shape
        dist2 = euclidean_dist_squared(X, means)
        dist2=np.array(dist2)
        sortIndex = np.argmin(dist2, axis=1)
        minDist2 = np.zeros(N)
        # print(dist2)
        for n in range(N):
            index = sortIndex[n:n+1]
            minDist2[n] = dist2[n,index]
        return np.sum(minDist2)
