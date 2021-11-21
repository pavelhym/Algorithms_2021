import numpy as np
from time import time
import matplotlib.pyplot as plt
import decimal
import pandas as pd
from collections import defaultdict
import copy



def random_adjacency_matrix_weighted(V,E):

    matrix = np.zeros((V,V))
    for n in range(E):
      i = np.random.randint(V)
      j = np.random.randint(V)
      while i == j or matrix[i][j] != 0:
        j = np.random.randint(V)
        i = np.random.randint(V)
      val = np.random.randint(100)
      matrix[i][j] = val
      matrix[j][i] = val
    return matrix



#Create adj matrix
adj_matrix = random_adjacency_matrix_weighted(100,500)
np.shape(adj_matrix)



def convert_to_adjacency_weighted(adj_matrix):
    graph = defaultdict(dict)
    edges = set()

    for i, v in enumerate(adj_matrix, 0):
        for j, u in enumerate(v, 0):
            if u != 0 :
                edges.add(frozenset([i, j]))
                graph[i].update({j: u})
    return graph





adj_list = convert_to_adjacency_weighted(adj_matrix)

adj_matrix_unweighted =  copy.deepcopy(adj_matrix)
adj_matrix_unweighted[adj_matrix_unweighted>0] = 1

#Graph

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd

adj_sparse = sp.sparse.coo_matrix(adj_matrix_unweighted, dtype=np.int8)
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


 
class Graph(object):
    def __init__(self, nodes, graph):
        self.nodes = nodes
        self.graph = graph

    
    def get_nodes(self):
        return self.nodes
    
    def neighbors(self, node):
        connections = []
        for out_node in self.nodes:
            if self.graph[node].get(out_node, False) != False:
                connections.append(out_node)
        return connections
    
    def dist(self, node1, node2):
        return self.graph[node1][node2]



def dijkstra_algorithm(graph, start_node):
    unvisited_nodes = list(graph.get_nodes())
 
       
    distance = {}
 
    previous_nodes = {}
 
    # Init dist 
    max_value = 1000000000000
    for node in unvisited_nodes:
        distance[node] = max_value
     
    distance[start_node] = 0
    
   
    while len(unvisited_nodes) > 0:
        # node with the lowest score
        current_node = None
        for node in unvisited_nodes: 
            if current_node == None:
                current_node = node
            elif distance[node] < distance[current_node]:
                current_node = node
                
        # Neighbors
        neighbors = graph.neighbors(current_node)
        
        for neighbor in neighbors:
            tentative_value = distance[current_node] + graph.dist(current_node, neighbor)
            if tentative_value < distance[neighbor]:
                distance[neighbor] = tentative_value
                
                previous_nodes[neighbor] = current_node
 
        
        unvisited_nodes.remove(current_node)
    
    return previous_nodes, distance




nodes = list(range(0,100))
graph_dijkstra = Graph(nodes, adj_list)
previous_nodes, shortest_path = dijkstra_algorithm(graph=graph_dijkstra, start_node=0)







def bellman_ford(graph, start_node):
   
    distance, previous_nodes = dict(), dict()
    for node in graph:
        distance[node], previous_nodes[node] = float('inf'), None
    distance[start_node] = 0

    for i in range(len(graph) - 1):
        for node in graph:
            for neighbour in graph[node]:
                
                if distance[neighbour] > distance[node] + graph[node][neighbour]:
                    distance[neighbour], previous_nodes[neighbour] = distance[node] + graph[node][neighbour], node

    for node in graph:
        for neighbour in graph[node]:
            assert distance[neighbour] <= distance[node] + graph[node][neighbour], "Negative weight cycle."
 
    return  previous_nodes, distance


distance, predecessor = bellman_ford(adj_list, start_node=0)




#calculate time

from timeit import Timer
import functools




iteration = list(range(0,10))
dijkstra = []
ford = []
for _ in range(10):
    start_node = np.random.randint(100)

    t = Timer(functools.partial(dijkstra_algorithm,graph_dijkstra,start_node))  
    time_eval = t.timeit(1)
    dijkstra.append(time_eval)

    t = Timer(functools.partial(bellman_ford,adj_list,start_node))  
    time_eval = t.timeit(1)
    ford.append(time_eval)



plt.plot(iteration, dijkstra, label = "dijkstra")
plt.plot(iteration, ford, label = 'bellman-ford')
plt.legend()
plt.title("Algorithms comparison")
plt.savefig('Plots/comparison.png')
plt.show()

print('AVG time for Dijkstra - ', np.mean(dijkstra))
print('AVG time for Bellman Ford - ', np.mean(ford))


#3


#to store info on each node
class Node():

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def A_star(graph, start, end, distance = "Euc" ,strict = False):

    
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Lists to store nodes to visit and visited nodes
    tovisit = []
    visited = []

    # Add the start node
    tovisit.append(start_node)

    # iterate over all fields
    while len(tovisit) > 0:

        
        current_node = tovisit[0]
        current_index = 0
        for index, item in enumerate(tovisit):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Visited
        
        tovisit.pop(current_index)
        visited.append(current_node)

        # Found the goal
        if current_node == end_node:
            print('Path is found')
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path

        # Possible new steps
        children = []
        if strict == True:
            reloc =  [(0, -1), (0, 1), (-1, 0), (1, 0)]
        else:
            reloc = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        for new_position in reloc: 

            # Expansion
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            
            if node_position[0] > (len(graph) - 1) or node_position[0] < 0 or node_position[1] > (len(graph[len(graph)-1]) -1) or node_position[1] < 0:
                continue

            #No obstacle codition
            if graph[node_position[0]][node_position[1]] != 0:
                continue

            # Possible expansions
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Choose next step
        for child in children:

            # Check visited or not
            skip = False
            for closed_child in visited:
                if child == closed_child:
                    skip= True
            if skip == True:
                continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            if distance == "Man":
                #Manhattan distance
                child.h = np.abs((child.position[0] - end_node.position[0])) + np.abs((child.position[1] - end_node.position[1]))
            else:
                #Euclidean distance
                child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)

            child.f = child.g + child.h
            
            skip = False
            
            for open_node in tovisit:
                if child == open_node and child.g > open_node.g:
                    skip= True
            if skip == True:
                continue

            
            tovisit.append(child)
    return print("Path does not exist")



def graph_with_obstacles(len,width,obstacles):

    matrix = np.zeros((len,width))
    for n in range(obstacles):
      i = np.random.randint(len)
      j = np.random.randint(width)
      while matrix[i][j] == 1:
        i = np.random.randint(len)
        j = np.random.randint(width)
      matrix[i][j] = 1 
    return matrix


obstacle_graph =  graph_with_obstacles(10,20,40)





start = (0, 0)
end = (5, 5)
path = A_star(obstacle_graph, start, end, "Man", strict=True)

astar = []
astar_man = []

for _ in range(5):
    start = (np.random.randint(10), np.random.randint(20))
    end = (np.random.randint(10), np.random.randint(20))
    while (start == end) or (obstacle_graph[start[0]][start[1]] == 1) or (obstacle_graph[end[0]][end[1]] == 1):
        start = (np.random.randint(10), np.random.randint(20))
        end = (np.random.randint(10), np.random.randint(20))

    #print(start)
    #print(end)
    t = Timer(functools.partial(A_star,obstacle_graph,start,end))  
    time_eval = t.timeit(1)
    astar.append(time_eval)

    t = Timer(functools.partial(A_star,obstacle_graph,start,end,"Man"))  
    time_eval = t.timeit(1)
    astar_man.append(time_eval)



plt.plot(list(range(0,5)), astar, label = "A* Euclidean time")
plt.plot(list(range(0,5)), astar_man, label = "A* Manhattan time")
plt.legend()
plt.title("A* time")
plt.savefig('Plots/Astar.png')
plt.show()

print("Mannhattan AVG", np.mean(astar_man))
print("Euc AVG", np.mean(astar))


astar100 = []
astar_man100 = []

for _ in range(100):
    start = (np.random.randint(10), np.random.randint(20))
    end = (np.random.randint(10), np.random.randint(20))
    while (start == end) or (obstacle_graph[start[0]][start[1]] == 1) or (obstacle_graph[end[0]][end[1]] == 1):
        start = (np.random.randint(10), np.random.randint(20))
        end = (np.random.randint(10), np.random.randint(20))

    #print(start)
    #print(end)
    t = Timer(functools.partial(A_star,obstacle_graph,start,end))  
    time_eval = t.timeit(1)
    astar100.append(time_eval)

    t = Timer(functools.partial(A_star,obstacle_graph,start,end,"Man"))  
    time_eval = t.timeit(1)
    astar_man100.append(time_eval)



plt.plot(list(range(0,100)), astar100, label = "A* Euclidean time")
plt.plot(list(range(0,100)), astar_man100, label = "A* Manhattan time")
plt.legend()
plt.title("A* time 100 iterations")
plt.savefig('Plots/Astar100.png')
plt.show()

print("Mannhattan AVG 100", np.mean(astar_man100))
print("Euc AVG 100", np.mean(astar100))



astar100_strict = []
astar_man100_strict = []

for _ in range(20):
    start = (np.random.randint(10), np.random.randint(20))
    end = (np.random.randint(10), np.random.randint(20))
    while (start == end) or (obstacle_graph[start[0]][start[1]] == 1) or (obstacle_graph[end[0]][end[1]] == 1):
        start = (np.random.randint(10), np.random.randint(20))
        end = (np.random.randint(10), np.random.randint(20))

    #print(start)
    #print(end)
    t = Timer(functools.partial(A_star,obstacle_graph,start,end,strict = True))  
    time_eval = t.timeit(1)
    astar100_strict.append(time_eval)

    t = Timer(functools.partial(A_star,obstacle_graph,start,end,"Man",strict = True))  
    time_eval = t.timeit(1)
    astar_man100_strict.append(time_eval)



plt.plot(list(range(0,20)), astar100_strict, label = "A* Euclidean time")
plt.plot(list(range(0,20)), astar_man100_strict, label = "A* Manhattan time")
plt.legend()
plt.title("A* time 100 iterations strict relocations")
plt.savefig('Plots/Astar100_strict.png')
plt.show()

print("Mannhattan AVG 100_strict", np.mean(astar_man100_strict))
print("Euc AVG 100_strict", np.mean(astar100_strict))