import numpy as np
from time import time
import matplotlib.pyplot as plt
import decimal
from numpy.core.fromnumeric import shape
import pandas as pd




def random_adjacency_matrix(V,E):

    matrix = np.zeros((V,V))
    for n in range(E):
      i = np.random.randint(V)
      j = np.random.randint(V)
      while i == j or matrix[i][j] == 1:
        j = np.random.randint(V)
        i = np.random.randint(V)
      matrix[i][j] = 1 
      matrix[j][i] = 1
    return matrix


adj_matrix =  random_adjacency_matrix(100,200)




def convert_to_adjacency(matrix):
    start = 0
    res = []
    lst = []
    n = len(matrix)

    for i in range(n):
        res.append(lst*n)
    while start < n:
        y = matrix[start]
        for i in range(len(y)):
            if y[i] == 1:
                res[start].append(i)
        start += 1
    return res

adj_list =  convert_to_adjacency(adj_matrix)

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd

adj_sparse = sp.sparse.coo_matrix(adj_matrix, dtype=np.int8)
labels = range(0,100)
DF_adj = pd.DataFrame(adj_sparse.toarray(),index=labels,columns=labels)
print(DF_adj)



#Network graph
G = nx.Graph()
G.add_nodes_from(labels)

#Connect nodes
for i in range(DF_adj.shape[0]):
    col_label = DF_adj.columns[i]
    for j in range(DF_adj.shape[1]):
        row_label = DF_adj.index[j]
        node = DF_adj.iloc[i,j]
        if node == 1:
            G.add_edge(col_label,row_label)


#Draw graph
nx.draw(G,with_labels = True)


#print matrix and list:

print(adj_matrix[:2])
print(adj_list[:2])




#2






visited = []



def dfs(visited, graph, node): 
    if node not in visited:
        #print(node)
        visited.append(node)
        for neighbour in graph[node]:
            dfs(visited, graph, neighbour)



def find_components(adj_list):
    visited = []
    grop_n = 0
    group = []
    components = [[]]
    for node in range(len(adj_list)):
        if node not in visited:

            dfs(group,adj_list,node)
            print(group)
            visited = visited + group
            components.append(group)
            group = []
            grop_n +=1
    components.pop(0)
    return grop_n, components

result = find_components(adj_list)




#BFS

start = 1
goal = 1
graph = adj_list


def BFS_SP(graph, start, goal):
    explored = []
     
    # Queue for traversing the
    # graph in the BFS
    queue = [[start]]
     
    # If the desired node is
    # reached
    if start == goal:
        print("Same Node")
        return
     
    # Loop to traverse the graph
    # with the help of the queue
    while queue:
        path = queue.pop(0)
        node = path[-1]
         
        # Condition to check if the
        # current node is not visited
        if node not in explored:
            neighbours = graph[node]
             
            # Loop to iterate over the
            # neighbours of the node
            for neighbour in neighbours:
                print(neighbour)
                new_path = list(path)
                new_path.append(neighbour)
                queue.append(new_path)
                 
                # Condition to check if the
                # neighbour node is the goal
                if neighbour == goal:
                    print("Shortest path = ", *new_path)
                    return
            explored.append(node)
 
    # Condition when the nodes
    # are not connected
    print("No connection")
    return


BFS_SP(adj_list,21,0)



