import time

import numpy as np


def gauss(A, b, n):
    y = np.empty(n, np.float)

    # loop over all rows
    for k in range(0, n):
        # parallelizable
        for j in range(k + 1, n):
            # compute multipliers
            A[k][j] = A[k][j] / A[k][k]
        y[k] = b[k] / A[k][k]
        A[k][k] = 1
        for i in range(k + 1, n):
            # update coefficients
            for j in range(k + 1, n):
                A[i][j] = A[i][j] - A[i][k] * A[k][j]
            b[i] = b[i] - A[i][k] * y[k]
            A[i][k] = 0

    x = np.zeros(n, np.float)
    # loop over all rows backwards
    for k in range(n - 1, -1, -1):
        x[k] = y[k]
        for i in range(k - 1, -1, -1):
            y[i] = y[i] - x[k] * A[i][k]
    return x


if __name__ == '__main__':
    # take input for an n value
    dimension = int(input("Enter dimension of nxn matrix:"))
    np.random.seed()
    pb = np.random.permutation(dimension)
    pA = np.random.rand(0, dimension, (dimension, dimension))
    start = time.time()
    gauss(pA, pb, dimension)
    end = time.time()
    print("Serial: On data of size ", dimension, " is ", end - start)
