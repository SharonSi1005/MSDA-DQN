# Python program for Dijkstra's single
# source shortest path algorithm. The program is
# for adjacency matrix representation of the graph
import numpy as np
import math


class Graph():

    def __init__(self, vertices):
        self.V = vertices
        self.graph = np.array([[0 for column in range(vertices)]
                               for row in range(vertices)])

    def printSolution(self, dist):
        print("Vertex \t Distance from Source")
        for node in range(self.V):
            print(node, "\t\t", dist[node])

    # A utility function to find the vertex with
    # minimum distance value, from the set of vertices
    # not yet included in shortest path tree
    # Line 11： 找到最短距离的node
    def minDistance(self, dist, sptSet):

        # Initialize minimum distance for next node
        min = 1e7

        # Search the nearest vertex not in the
        # shortest path tree
        for v in range(self.V):
            if dist[v] < min and sptSet[v] == False:
                min = dist[v]
                min_index = v

        return min_index

    # Function that implements Dijkstra's single source
    # shortest path algorithm for a graph represented
    # using adjacency matrix representation
    def dijkstra(self, src):
        # 参数初始化
        dist = [1e7] * self.V
        sptSet = [False] * self.V  # False: 没算的
        prev = [math.nan] * self.V

        dist[src] = 0

        for cout in range(self.V):

            # Pick the minimum distance vertex from
            # the set of vertices not yet processed.
            # u is always equal to src in first iteration
            u = self.minDistance(dist, sptSet)

            # Put the minimum distance vertex in the
            # shortest path tree
            sptSet[u] = True

            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the shortest path tree
            for v in range(self.V):
                if (self.graph[u, v] > 0.001 and  # 与Node u相连
                        sptSet[v] == False and  # 没算的node中
                        dist[v] > dist[u] + self.graph[u, v]):  # 判断比之前的距离短
                    dist[v] = dist[u] + self.graph[u, v]
                    prev[v] = u

        #  self.printSolution(dist)
        return dist, prev


if __name__ == "__main__":
    '''
    g = Graph(9)
    g.graph = np.array([[0, 4, 0, 0, 0, 0, 0, 8, 0],
                        [4, 0, 8, 0, 0, 0, 0, 11, 0],
                        [0, 8, 0, 7, 0, 4, 0, 0, 2],
                        [0, 0, 7, 0, 9, 14, 0, 0, 0],
                        [0, 0, 0, 9, 0, 10, 0, 0, 0],
                        [0, 0, 4, 14, 10, 0, 2, 0, 0],
                        [0, 0, 0, 0, 0, 2, 0, 1, 6],
                        [8, 11, 0, 0, 0, 0, 1, 0, 7],
                        [0, 0, 2, 0, 0, 0, 6, 7, 0]
                        ])
    print(g.graph)
    dist, prev = g.dijkstra(2)
    print(dist)
    print(prev)
    '''
    g = Graph(506)
    # 从 CSV 文件读取矩阵
    csv_filename = 'G_eqv_adj_matrix.csv'
    G_eqv_adj_matrix = np.loadtxt(csv_filename, delimiter=',')
    g.graph = G_eqv_adj_matrix
    print(g.graph)
    dist, prev = g.dijkstra(27)
    print(dist)
    print(prev)
    element = 129
    if element in prev:
        index = prev.index(element)
        print(f"元素 '{element}' 在列表中的索引为: {index}")
    else:
        print(f"元素 '{element}' 不在列表中。")
