"""
As an exercise, implement Dijkstra as described in "Cracking the Coding Interview" (6th edition page 644)
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# PriorityQueue naive implementation copied from: https://www.geeksforgeeks.org/priority-queue-in-python/.
# Adapted to have priorities in the poping function
class PriorityQueue(object):
    def __init__(self):
        self.data = []

    def __str__(self):
        return ' '.join([str(i) + ', ' for i in self.data])

    # for checking if the queue is empty
    def isEmpty(self):
        return len(self.data) == 0

    # for inserting an element in the queue
    def insert(self, data):
        self.data.append(data)

    # for popping an element based on Priority
    # I decided to give priorities as an input because one might wish to update it while poping
    def pop(self, priorities):
        try:
            ind_max_val = 0
            for i, p in enumerate(priorities):
                if p > priorities[ind_max_val]:
                    ind_max_val = i
            item = self.data[ind_max_val]
            del self.data[ind_max_val]
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
        remaining.insert(node)

    while not remaining.isEmpty():
        node = remaining.pop([-path_weight[d] for d in remaining.data])
        for neighbor in lil_graph[node]:
            weight_via_node = path_weight[node] + weights_matrix[node, neighbor]
            if path_weight[neighbor] > weight_via_node:
                path_weight[neighbor] = weight_via_node
                previous[neighbor] = node
    path_from_t_to_s = []
    node = t
    while node != s:
        path_from_t_to_s.append(node)
        node = previous[node]
    path_from_t_to_s.append(s)
    path_from_t_to_s.reverse()
    return path_from_t_to_s


def plot_network(lil_graph, weights, shortest_path):
    """
    https://networkx.org/documentation/stable/auto_examples/drawing/plot_weighted_graph.html
    """
    G = nx.DiGraph()
    for n in range(len(lil_graph)):
        for m in lil_graph[n]:
            G.add_edge(n, m, weight=weights[n, m])
    epath = [(shortest_path[i], shortest_path[i + 1]) for i in range(len(shortest_path) - 1)]
    pos = nx.spring_layout(G, seed=7)  # positions for all nodes - seed for reproducibility
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_edges(G, pos, edgelist=epath, width=2, edge_color='r')
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    # Simplest test
    lil_graph = [[1], [0]]
    weights = np.array([[-1, 1], [1, -1]])
    print(Dijkstra(lil_graph, weights, 0, 1))

    # More complicated test
    a, b, c, d, e, f, g, h, i = 0, 1, 2, 3, 4, 5, 6, 7, 8
    lil_graph = [[b, c], [d], [b, d], [a, g, h], [a, h, i], [b, g, i], [c, i], [f, g, c], []]
    weights = np.zeros((len(lil_graph), len(lil_graph)))
    for n in range(len(lil_graph)):
        for m in lil_graph[n]:
            weights[n, m] = np.random.randint(1, 10)
    shortest_path = Dijkstra(lil_graph, weights, a, i)
    plot_network(lil_graph, weights, shortest_path)


if __name__ == '__main__':
    main()
