from __future__ import division
import numpy as np

try:
    xrange
except NameError:
    xrange = range

def add_intercept(X_):
    m, n = X_.shape
    X = np.zeros((m, n + 1))
    X[:, 0] = 1
    X[:, 1:] = X_
    return X

def load_data(filename):
    D = np.loadtxt(filename)
    Y = D[:, 0]
    X = D[:, 1:]
    return add_intercept(X), Y

def calc_grad(X, Y, theta):
    # Vectorized computation of the gradient of the log likelihood function
    m, n = X.shape
    grad = np.zeros(theta.shape)

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1./m) * (X.T.dot(probs * Y))

    return grad

def logistic_regression(X, Y):
    # Input:
    #    X ... Features
    #    Y ... Labels in {-1, +1)
    #          Note: The version we saw in class assumed labels in {0,1}
    #                (The gradient formula here is therefore slightly different)
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 0.000000000000001

    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta  - learning_rate * (grad)
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
    return

def main():
    print('==== Training model on data set A ====')
    Xa, Ya = load_data('hw3_data_a.txt')
    logistic_regression(Xa, Ya)

    print('\n==== Training model on data set B ====')
    Xb, Yb = load_data('hw3_data_b.txt')
    logistic_regression(Xb, Yb)

    return

if __name__ == '__main__':
    main()
