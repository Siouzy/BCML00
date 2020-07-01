import numpy as np
from tools import add_intercept


def predict_(x, theta):
    theta = theta.reshape((1, -1))
    x = add_intercept(x)
    y = x * theta
    return y.sum(axis=1)
