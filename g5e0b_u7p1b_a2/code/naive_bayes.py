import numpy as np

class NaiveBayes:
    # Naive Bayes implementation.
    # Assumes the feature are binary.
    # Also assumes the labels go from 0,1,...C-1

    def __init__(self, num_classes, beta=0):
        self.num_classes = num_classes
        self.beta = beta

    def fit(self, X, y):
        N, D = X.shape

        # Compute the number of class labels
        C = self.num_classes

        # Compute the probability of each class i.e p(y==c)
        counts = np.bincount(y)
        p_y = counts / N

        # Compute the conditional probabilities i.e.
        # p(x(i,j)=1 | y(i)==c) as p_xy

        # p_xy = 0.5 * np.ones((D, C))

        p_xy_test_smaller = np.zeros((4, D))

        beta = self.beta

        column_index = 0
        for column in X.T:
            for index, column_val in enumerate(column):
                if y[index] == 0:
                    if column_val == 1:
                        p_xy_test_smaller[0, column_index] += 1
                elif y[index] == 1:
                    if column_val == 1:
                        p_xy_test_smaller[1, column_index] += 1
                elif y[index] == 2:
                    if column_val == 1:
                        p_xy_test_smaller[2, column_index] += 1
                elif y[index] == 3:
                    if column_val == 1:
                        p_xy_test_smaller[3, column_index] += 1

            column_index += 1

        transposedArray = p_xy_test_smaller.T

        # Question 1.4
        transposedArray += beta

        transposedArray[:, 0] /= counts[0]
        transposedArray[:, 1] /= counts[1]
        transposedArray[:, 2] /= counts[2]
        transposedArray[:, 3] /= counts[3]

        p_xy = transposedArray

        self.p_y = p_y
        self.p_xy = p_xy


    # This function is provided just for your understanding.
    # It should function the same as predict()
    def predict(self, X):

        N, D = X.shape
        C = self.num_classes
        p_xy = self.p_xy
        p_y = self.p_y

        y_pred = np.zeros(N)
        for n in range(N):

            probs = p_y.copy() # initialize with the p(y) terms
            for d in range(D):
                if X[n, d] != 0:
                    probs *= p_xy[d, :]
                else:
                    probs *= (1-p_xy[d, :])

            y_pred[n] = np.argmax(probs)

        return y_pred
