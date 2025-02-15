import numpy as np


def simple_predict(x, theta):
    y = []
    for val in x:
        y.append(theta[0] + theta[1] * val)
    return np.asarray(y)


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
