"""
As an exercise, implement Dijkstra as described in "Cracking the Coding Interview" (6th edition page 644)
"""

import numpy as np


# PriorityQueue naive implementation copied from: https://www.geeksforgeeks.org/priority-queue-in-python/.
# Adapted to have both index and priority
class PriorityQueue(object):
    def __init__(self):
        self.data = []
        self.priority = []

    def __str__(self):
        return ' '.join([str(self.data[i]) + ',' + str(self.priority[i]) + '; ' for i in range(len(self.data))])

    # for checking if the queue is empty
    def isEmpty(self):
        return len(self.data) == 0

    # for inserting an element in the queue
    def insert(self, data, priority):
        self.data.append(data)
        self.priority.append(priority)

    # for popping an element based on Priority
    def pop(self):
        try:
            ind_max_val = 0
            for i in range(len(self.data)):
                if self.priority[i] > self.priority[ind_max_val]:
                    ind_max_val = i
            item = self.data[ind_max_val]
            del self.data[ind_max_val]
            del self.priority[ind_max_val]
            return item
        except IndexError:
            print()
            exit()


def Dijkstra(lil_graph, weights_matrix, s, t):
    """
    Assumes the weighted digraph is given by a matrix structure.
    Dijkstra algorithm find the shortest path from s to any other node in the graph. In particular, we use it to find
    the shortest path from s to t.
    :param lil_graph: list of lists graph representation. lil_graph[a] contain node a's neighbors.
    :param weights_matrix: graph weights matrix.
    :param s: Starting node. int between 0 to len(lil_graph)
    :param t: Last node. int between 0 to len(lil_graph)
    :return: path from s to t, that is a list of nodes that starts with s and ends with t.
    """
    n = len(weights_matrix)
    path_weight = [np.inf for _ in range(n)]
    path_weight[s] = 0
    previous = [-1 for _ in range(n)]
    remaining = PriorityQueue()
    for node in range(n):
        remaining.insert(node, path_weight[node])

    while not remaining.isEmpty():
        node = remaining.pop()
        for neighbor in lil_graph[node]:
            weight_via_node = path_weight[node] + weights_matrix[node, neighbor]
            if path_weight[neighbor] > weight_via_node:
                path_weight[neighbor] = weight_via_node
                previous[neighbor] = node
    # Big bug! update path_weight vector does not update remaining priorities. Not sure how to fix it yet.

