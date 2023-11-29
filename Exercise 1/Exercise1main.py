# Imports
import numpy as np

# Global Variables
ALPHA = 0.
BETA = -1.


# Create Matrix
def create_huckel_matrix(n=4, mode='linear'):
    """
    Creates a huckel matrix based off the length and mode inputted
    :param int n: length of polyene if linear or cyclic
    :param str mode: type of matrix to be generated
    :return array matrix: returns the matrix
    """
    matrix = np.zeros((n, n))

    for i in range(n-1):
        matrix[i, i+1] = BETA
        matrix[i+1, i] = BETA
        matrix[i+1, i+1] = ALPHA
    matrix[0, 0] = ALPHA

    if mode == 'cyclic':
        matrix[n-1, 0], matrix[0, n-1] = BETA, BETA

    elif mode == 'cube':
        matrix = -np.asarray([[0, 1, 0, 1, 1, 0, 0, 0],
                              [1, 0, 1, 0, 0, 1, 0, 0],
                              [0, 1, 0, 1, 0, 0, 1, 0],
                              [1, 0, 1, 0, 0, 0, 0, 1],
                              [1, 0, 0, 0, 0, 1, 0, 1],
                              [0, 1, 0, 0, 1, 0, 1, 0],
                              [0, 0, 1, 0, 0, 1, 0, 1],
                              [0, 0, 0, 1, 1, 0, 1, 0]])

    elif mode == 'tetrahedron':
        matrix = -np.asarray([[0, 1, 1, 1],
                              [1, 0, 1, 1],
                              [1, 1, 0, 1],
                              [1, 1, 1, 0]])

    elif mode == 'dodecahedron':
        matrix = -np.asarray([[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]])

    elif mode == 'buckyball':
        matrix = -np.loadtxt(open("bucky.csv", "rb"), delimiter=",")

    return matrix


def get_evals(matrix):
    """Return the eigenvalues of the matrix"""
    eigenvals = np.linalg.eigvals(matrix)
    return eigenvals


def get_input():
    """
    Gets the desired length for linear and cyclic polyenes
    :return int n: integer length
    """
    n = input('n: ')
    try:
        n = int(n)
        if n <= 0:
            print('invalid input')
            n = get_input()
    except ValueError:
        print('invalid input')
        n = get_input()
    return n


def get_type():
    """
    Gets the type of huckel matrix to be calculated
    :return str input_mode: the mode of matrix to be calculated
    """
    input_mode = input('Mode (cyclic, linear, tetrahedron, cube, dodecahedron or buckyball): ')
    if input_mode not in ['cyclic', 'linear', 'cube', 'tetrahedron', 'dodecahedron', 'buckyball']:
        input_mode = get_type()
    return input_mode


def count_degeneracies(eigenvals):
    """
    Counts and prints the degeneracies of the eigenvalues
    :param list eigenvals: list of all eigenvalues (floats)
    :return:
    """
    degeneracies = []

    for i, a in enumerate(eigenvals):
        eigenvals[i] = a.round(9)
    unique_eigenvals = np.unique(eigenvals)
    for eigenval in unique_eigenvals:
        degen = eigenvals.count(eigenval)
        degeneracies.append((eigenval, degen))

    degeneracies = sorted(degeneracies)
    for energy, degeneracy in degeneracies:
        print(f'Energy: α{-1*energy.real:+}β, Degeneracy: {degeneracy}')
    print('')


if __name__ == "__main__":
    try:
        while True:
            molecule = get_type()
            size = get_input() if molecule in ['cyclic', 'linear'] else 1
            print('')
            master_matrix = create_huckel_matrix(size, mode=molecule)
            evals = list(get_evals(master_matrix))
            count_degeneracies(evals)
    except KeyboardInterrupt:
        print("\nRUNTIME TERMINATED SUCCESSFULLY")
