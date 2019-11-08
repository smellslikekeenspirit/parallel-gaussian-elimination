import time

import numpy as np

import multiprocessing as mp

"""
n = dimension of matrix
A = nxn matrix input by user
b = vector of length n
y = augmented vector after forward elimination
x = solution for the system of linear equations
"""


def compute_multipliers(args):
    A, k, j = args
    A[k][j] = A[k][j] / A[k][k]
    return


def update_helper(A, i, k, n):
    for j in range(k + 1, n):
        A[i][j] = A[i][j] - A[i][k] * A[k][j]
    return


def update_coefficients(args):
    A, b, y, k, i, n = args
    update_helper(A, i, k, n)
    b[i] = b[i] - A[i][k] * y[k]
    A[i][k] = 0
    return


def forward_elimination(A, b, y, n):
    for k in range(0, n):
        # A and k remain unchanged over the repetitions while j changes
        args = [(A, k, j) for j in range(k + 1, n)]
        # pool of workers created, sectioned into cores
        pool1 = mp.Pool(mp.cpu_count())
        # computations for j = k+1 to n
        pool1.map(compute_multipliers, args)
        pool1.close()
        y[k] = b[k] / A[k][k]
        A[k][k] = 1
        pool1.join()
        args = [(A, b, y, k, i, n) for i in range(k + 1, n)]
        pool2 = mp.pool.Pool(mp.cpu_count())
        pool2.map(update_coefficients, args)
        pool2.close()
        pool2.join()
    return A, y


def back_substitution(U, y, n):
    x = np.zeros(n, np.float)
    # loop over all rows backwards
    for k in range(n - 1, -1, -1):
        x[k] = y[k]
        # loop backwards and update
        for i in range(k - 1, -1, -1):
            y[i] = y[i] - x[k] * U[i][k]
    return x


def gauss(A, b, dimension):
    # performs the two parts of solving equations
    y = np.empty(dimension, np.float)
    # forward elimination
    augA, augY = forward_elimination(A, b, y, dimension)
    # back substitution
    x = back_substitution(augA, augY, dimension)
    return x


if __name__ == '__main__':
    #taking n from user input
    dimension = int(input("Enter dimension of nxn matrix:"))
    # generating random seed
    np.random.seed()
    pb = np.random.permutation(dimension)
    # n x n array of random numbers
    pA = np.random.randint(1, dimension * 10, (dimension, dimension))
    start = time.time()
    gauss(pA, pb, dimension)
    end = time.time()
    print("Parallel: On data of size ", dimension, " is ", end-start)

