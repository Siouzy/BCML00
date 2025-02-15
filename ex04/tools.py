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


if __name__ == "__main__":
    import numpy as np
    x = np.arange(1, 6)
    print(add_intercept(x))
    print(np.array([[1., 1.],
                    [1., 2.],
                    [1., 3.],
                    [1., 4.],
                    [1., 5.]]
                   ))
    y = np.arange(1, 10).reshape((3, 3))
    print(add_intercept(y))
    print(np.array([[1., 1., 2., 3.],
                    [1., 4., 5., 6.],
                    [1., 7., 8., 9.]]
                   ))
