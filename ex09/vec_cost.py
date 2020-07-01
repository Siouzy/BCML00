import numpy as np


def cost_(y, y_hat):
    dif = y - y_hat
    prod = np.dot(dif, dif)
    m = y.shape[0]
    return prod / (2 * m)
