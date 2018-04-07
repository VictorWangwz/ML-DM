import numpy as np
import decision_stump


def predict(X):
    N, D = X.shape
    y = np.zeros(N)

    for n in range(N):
        if X[n, 1] > 37.669007:
            if X[n, 0] > -96.090109:
                y[n] = 1
            else:
                y[n] = 2
        else:
            if X[n, 0] > -115.577574:
                y[n] = 2
            else:
                y[n] = 1

    return y
