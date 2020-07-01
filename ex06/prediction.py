import numpy as np
from tools import add_intercept


def predict_(x, theta):
    x = add_intercept(x)
    y = x * theta
    return y.sum(axis=1)
