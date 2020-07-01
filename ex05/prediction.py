import numpy as np
from tools import add_intercept


def predict_(x, theta):
    x = add_intercept(x)
    y = x * theta
    return y.sum(axis=1)


if __name__ == "__main__":
    x = np.arange(1, 6)
    theta1 = np.array([5, 0])
    print(simple_predict(x, theta1))
    print(np.array([5., 5., 5., 5., 5.]))
    theta2 = np.array([0, 1])
    print(simple_predict(x, theta2))
    print(np.array([1., 2., 3., 4., 5.]))
    theta3 = np.array([5, 3])
    print(simple_predict(x, theta3))
    print(np.array([8., 11., 14., 17., 20.]))
    theta4 = np.array([-3, 1])
    print(simple_predict(x, theta4))
    print(np.array([-2., -1.,  0.,  1.,  2.]))
