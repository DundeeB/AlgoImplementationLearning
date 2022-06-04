"""
https://en.wikipedia.org/wiki/Dynamic_programming
https://en.wikipedia.org/wiki/Matrix_chain_multiplication
"""

import numpy as np


def recursive_matrix_multipication(matrix_list):
    """
    Solve the matrix multiplication problem: given the series of matrix A_0,A_1,...,A_(n-1), we wish to calculate their
    multiplication A_0*A_1*...*A_(n-1). Since matrix multiplication is associative, we can do so in many ways, that is
    locate the brackets in many positions. The cost of multiplying n*m matrix with an m*l matrix is n*m*l, so there is
    a choice which is most efficient. This function finds the cost of the most efficient choice, and perform the
    multiplication.
    :param matrix_dims: list of integers. The i'th matrix has dimensions matrix_dims[i], matrix_dims[i+1].
    :return: The cost of the optimal solution, the matrix product
    """
    matrix_dims = [len(A) for A in matrix_list] + [matrix_list[-1].shape[1]]
    # The costs matrix: costs[i,j] is the optimal cost to multiply A_i*A_(i+1)*...*A_j
    n = len(matrix_dims) - 1
    costs = np.zeros((n, n), dtype=int)
    costs[:] = -1
    for i in range(n): costs[i, i] = 0
    optimal_bracket_choice = np.zeros((n, n), dtype=int)
    optimal_bracket_choice[:] = -1

    def calc_cost(i, j):
        # Calculate the cost of multiplying matrices A_i*..*A_j
        if costs[i, j] != -1:
            return costs[i, j]
        costs_per_k = np.zeros(j - i)
        for k in range(i, j):  # k = i...(j-1)
            # Calculate (A_i*...*A_k)*(A_(k+1)*...*A_j)
            costs_per_k[k - i] = calc_cost(i, k) + calc_cost(k + 1, j) + \
                                 matrix_dims[i] * matrix_dims[k + 1] * matrix_dims[j + 1]
            # The cost of breaking an expression to two expressions is the cost of each subexpression + cost of final
            # multiplication.
        optimal_bracket_choice[i, j] = np.argmin(costs_per_k) + i
        costs[i, j] = costs_per_k[optimal_bracket_choice[i, j] - i]
        return costs[i, j]

    optimal_cost = calc_cost(0, n - 1)

    # Also this line fills the needed values in optimal_bracket_choice

    def multiply(i, j):
        if i == j:
            return matrix_list[i]
        elif j == i + 1:
            return np.matmul(matrix_list[i], matrix_list[j])
        k = optimal_bracket_choice[i, j]
        return np.matmul(multiply(i, k), multiply(k + 1, j))

    return optimal_cost, multiply(0, n - 1)


def main():
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    C = np.array([[1, 2], [4, 5], [7, 8]])
    cost, product = recursive_matrix_multipication([A, B, C])
    assert (product == np.matmul(np.matmul(A, B), C)).all(), "Product is incorrect!"
    print("cost: " + str(cost))
    print("product: " + str(product))

if __name__ == "__main__":
    main()
