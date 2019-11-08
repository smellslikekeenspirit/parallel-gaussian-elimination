import copy
import multiprocessing

import gaussian_elimination_seq as gs
import gaussian_elimination_par as gp
import numpy as np
import time
import nested_pool as nest

def gauss_par(args):
    pA, pb, dimension, i = args
    start = time.time()
    gs.gauss(pA, pb, dimension)
    end = time.time()
    return end - start

if __name__ == '__main__':
    dimension = int(input("Enter dimension of nxn matrix:"))
    np.random.seed()
    pb = np.random.permutation(dimension)
    pA = np.random.randint(0, dimension, (dimension, dimension))
    sA = copy.deepcopy(pA)
    sb = copy.deepcopy(pb)
    print(pA)
    print(pb)

    time_gp = 0
    pool = nest.MyPool(multiprocessing.cpu_count())
    args = [(pA, pb, dimension, i) for i in range(0, 10)]
    results = pool.map(gauss_par, args)
    for r in results:
        time_gp += r
    time_gp /= 10
    pool.close()
    print("Parallel: Average time recorded for 10 runs on data of size ", dimension, " is ", time_gp)

    time_gs = 0
    for i in range(0, 10):
        start = time.time()
        gs.gauss(sA, sb, dimension)
        end = time.time()
        time_gs += end - start
    time_gs /= 10
    print("Serial: Average time recorded for 10 runs on data of size ", dimension, " is ", time_gs)






