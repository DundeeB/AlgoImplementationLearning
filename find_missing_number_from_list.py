"""
06/06/2022 interview IBEX, Jonathan Harael demo interview.
"""

import numpy as np

"""
Assuming a list of integers, such as when list length is n it includes uniquely all the numbers 1...(n+1) except one
"""


def find_missing_number_with_sorting(list_of_integers):
    np_integers = np.sort(list_of_integers)  # O(n*log(n))
    for i in range(len(np_integers)):
        if i + 1 != np_integers[i]:
            return i + 1
    return len(np_integers) + 1


def find_missing_number(list_of_integers):
    sum_of_list = 0
    for member_in_list in list_of_integers:
        sum_of_list += member_in_list

    sum_with_all_numbers = 0
    for i in range(1, len(list_of_integers) + 2):
        sum_with_all_numbers += i

    return sum_with_all_numbers - sum_of_list
