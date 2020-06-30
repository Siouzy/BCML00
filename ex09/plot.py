import numpy as np
import matplotlib.pyplot as plt
from prediction import predict_
from vec_cost import cost_

def plot_with_cost(x, y, theta):
    plt.title("Cost : " + str(cost_(y, predict_(x, theta))))
    plt.plot(x, y, 'ro', color='blue')
    pred = predict_(x, theta)
    plt.plot(x, pred, color='orange')
    zeroes = np.zeros_like(x)
    difs = np.concatenate((zeroes, x))
    for valx, valpred, valy in zip(x, pred, y):
        if valpred > valy:
            mini = valy
            maxi = valpred
        else:
            mini = valpred
            maxi = valy
        plt.vlines(x=valx, ymin=mini, ymax=maxi, linestyle='--', color='red')
    plt.show()


if __name__ == "__main__":
    x = np.arange(1,6)
    y = np.array([11.52434424, 10.62589482, 13.14755699, 18.60682298, 14.14329568])
    #Example 1:
    theta1= np.array([18,-1])
    plot_with_cost(x, y, theta1)
    #Example 2:
    theta2 = np.array([14, 0])
    plot_with_cost(x, y, theta2)
    #Example 3:
    theta3 = np.array([12, 0.8])
    plot_with_cost(x, y, theta3)