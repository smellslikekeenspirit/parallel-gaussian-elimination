import multiprocessing as mp
import gaussian_elimination_par


def parse(dimension):
    matrix = list()
    vector = list()
    for i in range(0, dimension):
        row = [int(y) for y in (input("Enter Row " + str(i + 1) + ":")).split(' ')]
        matrix.append(row)
    for i in range(0, dimension):
        vector.append(int(input("Enter vector element " + str(i + 1) + ":")))
    return matrix, vector


def main():
    matrix, vector = parse()




