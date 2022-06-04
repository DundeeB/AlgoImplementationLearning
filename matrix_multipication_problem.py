"""
https://en.wikipedia.org/wiki/Dynamic_programming
https://en.wikipedia.org/wiki/Matrix_chain_multiplication
"""

import numpy as np


def recursive_matrix_multipication_cost(matrix_dims):
    """
    Solve the matrix multiplication problem: given the series of matrix A_0,A_1,...,A_(n-1), we wish to calculate their
    multiplication A_0*A_1*...*A_(n-1). Since matrix multiplication is associative, we can do so in many ways, that is
    locate the brackets in many positions. The cost of multiplying n*m matrix with an m*l matrix is n*m*l, so there is
    a choices which is most efficient. This function finds the cost of the most efficient choice, although it does not
    tell you what the choice is.
    :param matrix_dims: list of integers. The i'th matrix has dimensions matrix_dims[i], matrix_dims[i+1].
    :return: The cost of the optimal solution.
    """

    # The costs matrix: costs[i,j] is the optimal cost to multiply A_i*A_(i+1)*...*A_j
    n = len(matrix_dims) - 1
    costs = np.zeros((n, n))
    costs[:] = np.nan
    for i in range(n): costs[i, i] = 0

    def recursive_cost_calculation(i, j):
        # Calculate the cost of multiplying matrices A_i*..*A_j
        if not np.isnan(costs[i, j]):
            return costs[i, j]
        costs_per_k = np.zeros(j - i)
        for k in range(i, j):  # k = i...(j-1)
            # Calculate (A_i*...*A_k)*(A_(k+1)*...*A_j)
            costs_per_k[k - i] = recursive_cost_calculation(i, k) + recursive_cost_calculation(k + 1, j) + \
                                 matrix_dims[i] * matrix_dims[k + 1] * matrix_dims[j + 1]
            # The cost of breaking an expression to two expressions is the cost of each subexpression + cost of final
            # multiplication.
        optimal_k = np.argmin(costs_per_k) + i
        costs[i, j] = costs_per_k[optimal_k - i]
        return costs[i, j]

    return recursive_cost_calculation(0, n - 1)
