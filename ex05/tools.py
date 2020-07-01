import numpy as np


def add_intercept(x):
    ones = np.ones((x.shape[0], 1))
    try:
        y = np.concatenate((ones, x), axis=1)
        return y
    except ValueError:
        x = x.reshape(-1, 1)
        y = np.concatenate((ones, x), axis=1)
        return y
