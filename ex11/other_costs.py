import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt


def mse_(y, y_hat):
    m = y.shape[0]
    dif = y_hat - y
    sigma = np.array(list(map(lambda x: x ** 2, dif)))
    return sigma.sum(axis=0) / m


def rmse_(y, y_hat):
    return sqrt(mse_(y, y_hat))


def mae_(y, y_hat):
    m = y.shape[0]
    dif = y_hat - y
    sigma = np.array(list(map(lambda x: abs(x), dif)))
    return sigma.sum(axis=0) / m


def r2score_(y, y_hat):
    dif1 = y_hat - y
    y_mean = np.full(y.shape, np.mean(y))
    dif2 = y_hat - y_mean
    sigma1 = np.array(list(map(lambda x: x ** 2, dif1)))
    sigma2 = np.array(list(map(lambda x: x ** 2, dif2)))
    return 1 - (sigma1.sum(axis=0) / sigma2.sum(axis=0))


if __name__ == "__main__":
    x = np.array([0, 15, -9, 7, 12, 3, -21])
    y = np.array([2, 14, -13, 5, 12, 4, -19])
    # Mean squared error
    # your implementation
    print(mse_(x, y))
    # sklearn implementation
    print(mean_squared_error(x, y))
    # Root mean squared error
    # your implementation
    print(rmse_(x, y))
    # sklearn implementation not available: take the square root of MSE
    print(sqrt(mean_squared_error(x, y)))
    # Mean absolute error
    # your implementation
    print(mae_(x, y))
    # sklearn implementation
    print(mean_absolute_error(x, y))
    # R2-score
    # your implementation
    print(r2score_(x, y))
    # sklearn implementation
    print(r2_score(x, y))
