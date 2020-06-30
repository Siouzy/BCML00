import numpy as np

def cost_(y, y_hat):
    dif = y - y_hat
    prod = np.dot(dif, dif)
    m = y.shape[0]
    return prod / (2 * m)

if __name__ == "__main__":
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    Y = np.array([2, 14, -13, 5, 12, 4, -19])
    # Example 1:
    print(cost_(X, Y))
    # Output:
    print(2.142857142857143)
    # Example 2:
    print(cost_(X, X))
    # Output:
    print(0.0)