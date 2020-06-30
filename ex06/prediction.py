import numpy as np
from tools import add_intercept

def predict_(x, theta):
    x = add_intercept(x)
    y = x * theta
    return y.sum(axis=1)

if __name__ == "__main__":
    x = np.arange(1,6)
    #Example 1:
    theta1 = np.array([5, 0])
    print(predict_(x, theta1))
    # Ouput:
    print(np.array([5., 5., 5., 5., 5.]))
    # Do you understand why y_hat contains only 5's here?
    #Example 2:
    theta2 = np.array([0, 1])
    print(predict_(x, theta2))
    # Output:
    print(np.array([1., 2., 3., 4., 5.]))
    # Do you understand why y_hat == x here?
    #Example 3:
    theta3 = np.array([5, 3])
    print(predict_(x, theta3))
    # Output:
    print(np.array([ 8., 11., 14., 17., 20.]))
    #Example 4:
    theta4 = np.array([-3, 1])
    print(predict_(x, theta4))
    # Output:
    print(np.array([-2., -1.,  0.,  1.,  2.]))