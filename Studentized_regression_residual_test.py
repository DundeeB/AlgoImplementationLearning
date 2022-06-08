"""
To demonstrate a project I did in my previous work, I re-implement it in python. Our goal is given a independent
variable x and dependent variable y, determine whether we have an outlier in the data. The emphasise is not on cleaning
the outlier, but rather calculate its p-value. It is done by generalizing Grubbs's test
(https://en.wikipedia.org/wiki/Grubbs%27s_test) to the studentized residuals
(https://en.wikipedia.org/wiki/Studentized_residual). Notice: since the exact pdf of the studentized residuals depends
on the design matrix, we use the monte-carlo method to numerically calculate the pdf.
"""

import numpy as np
from os.path import exists
import matplotlib.pyplot as plt
import hashlib


def simplest_design_matrix(x_vec):
    return np.array([np.ones(len(x_vec)), x_vec]).T


def hat_matrix(design_matrix):
    X = design_matrix
    return X @ np.linalg.inv(X.T @ X) @ X.T


def calc_influence(design_matrix):
    H = hat_matrix(design_matrix)
    return np.array([np.sqrt(1 - H[i, i]) for i in range(len(H))])


def std_estimator(residuals, dof):
    # dof = degrees of freedom
    std_square = (1 / (len(residuals) - dof)) * np.sum(np.square(residuals))
    return np.sqrt(std_square)


def studentized(residuals, influence, dof):
    sig = std_estimator(residuals, dof)
    t = residuals / (sig * influence)
    return t


def perform_monte_carlo(design_matrix, realizations=int(1e4), write_to_file=True, read_from_file=True,
                        visulize_T_pdf=False):
    X = design_matrix
    n, dof = X.shape
    hash_obj = hashlib.sha256()
    hash_obj.update(bytes(str(X), 'utf-8'))
    hash_str = hash_obj.hexdigest()[:3]
    MC_T_pdf_file_name = 'MC_T_pdf_realization=' + str(realizations) + '_X-hash=' + hash_str + '.txt'
    if read_from_file and exists(MC_T_pdf_file_name):
        T = np.loadtxt(MC_T_pdf_file_name)
    else:
        influence = calc_influence(X)  # no need to recalculate the influence every realization
        T = np.zeros(realizations)
        for i in range(realizations):
            residuals = np.random.normal(0, 1, n)
            t = studentized(residuals, influence, dof)
            T[i] = np.max(np.abs(t))
        T = np.sort(T)
        if write_to_file:
            np.savetxt(MC_T_pdf_file_name, T.T)
    if visulize_T_pdf:
        hist, bin_edges = np.histogram(T, 20, density=True)
        plt.plot([(bin_edges[i] + bin_edges[i + 1])/2 for i in range(len(bin_edges) - 1)], hist, 'o')
        plt.xlabel('$T=max(t_i)$, for $n=' + str(n) + '$')
        plt.ylabel('PDF')
    return T


def is_outlier(design_matrix, dependent_variable, alpha, realizations=int(1e4), visualize=True):
    X, y = design_matrix, dependent_variable
    n, dof = X.shape
    T = perform_monte_carlo(X, realizations, visulize_T_pdf=visualize)
    residuals_of_y = (np.identity(n) - hat_matrix(X)) @ y.T
    T_of_y = np.max(np.abs(studentized(residuals=residuals_of_y, influence=calc_influence(X), dof=dof)))
    index_in_T = np.where(T_of_y < T)[0][0]
    if visualize:
        plt.plot(2 * [T_of_y], [0, 1])
        plt.show()
    p_value = 1 - index_in_T / len(T)
    return 1 - p_value > alpha, p_value


def main():
    n = 10
    design_matrix = simplest_design_matrix([x for x in range(n)])
    dependent_variable = np.array(range(n)) + np.random.normal(0, 1, n)
    dependent_variable[int(n / 2)] += 6
    print(is_outlier(design_matrix, dependent_variable, 0.95, realizations=int(1e5)))


if __name__ == "__main__":
    main()
