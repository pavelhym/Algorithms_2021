import numpy as np
from time import time
import matplotlib.pyplot as plt
import decimal
import pandas as pd
from collections import defaultdict




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


import sys
 
class Graph(object):
    def __init__(self, nodes, graph):
        self.nodes = nodes
        self.graph = graph

    
    def get_nodes(self):
        "Returns the nodes of the graph."
        return self.nodes
    
    def get_outgoing_edges(self, node):
        "Returns the neighbors of a node."
        connections = []
        for out_node in self.nodes:
            if self.graph[node].get(out_node, False) != False:
                connections.append(out_node)
        return connections
    
    def value(self, node1, node2):
        "Returns the value of an edge between two nodes."
        return self.graph[node1][node2]



 


def dijkstra_algorithm(graph, start_node):
    unvisited_nodes = list(graph.get_nodes())
 
    # We'll use this dict to save the cost of visiting each node and update it as we move along the graph   
    distance = {}
 
    # We'll use this dict to save the shortest known path to a node found so far
    previous_nodes = {}
 
    # We'll use max_value to initialize the "infinity" value of the unvisited nodes   
    max_value = sys.maxsize
    for node in unvisited_nodes:
        distance[node] = max_value
    # However, we initialize the starting node's value with 0   
    distance[start_node] = 0
    
    # The algorithm executes until we visit all nodes
    while unvisited_nodes:
        # The code block below finds the node with the lowest score
        current_min_node = None
        for node in unvisited_nodes: # Iterate over the nodes
            if current_min_node == None:
                current_min_node = node
            elif distance[node] < distance[current_min_node]:
                current_min_node = node
                
        # The code block below retrieves the current node's neighbors and updates their distances
        neighbors = graph.get_outgoing_edges(current_min_node)
        for neighbor in neighbors:
            tentative_value = distance[current_min_node] + graph.value(current_min_node, neighbor)
            if tentative_value < distance[neighbor]:
                distance[neighbor] = tentative_value
                # We also update the best path to the current node
                previous_nodes[neighbor] = current_min_node
 
        # After visiting its neighbors, we mark the node as "visited"
        unvisited_nodes.remove(current_min_node)
    
    return previous_nodes, distance




nodes = list(range(0,100))
graph_dijkstra = Graph(nodes, adj_list)
previous_nodes, shortest_path = dijkstra_algorithm(graph=graph_dijkstra, start_node=0)







def bellman_ford(graph, start_node):
    # Step 1: Prepare the distance and previous_nodes for each node
    distance, previous_nodes = dict(), dict()
    for node in graph:
        distance[node], previous_nodes[node] = float('inf'), None
    distance[start_node] = 0

    # Step 2: Relax the edges
    for _ in range(len(graph) - 1):
        for node in graph:
            for neighbour in graph[node]:
                # If the distance between the node and the neighbour is lower than the current, store it
                if distance[neighbour] > distance[node] + graph[node][neighbour]:
                    distance[neighbour], previous_nodes[neighbour] = distance[node] + graph[node][neighbour], node

    # Step 3: Check for negative weight cycles
    for node in graph:
        for neighbour in graph[node]:
            assert distance[neighbour] <= distance[node] + graph[node][neighbour], "Negative weight cycle."
 
    return  previous_nodes, distance


distance, predecessor = bellman_ford(adj_list, start_node=0)




#calculate time

from timeit import Timer
import functools





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


def A_star(graph, start, end):

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Lists to store nodes to visit and visited nodes
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Come till the end
    while len(open_list) > 0:

        # Take first node from open list (then it will be the first child)
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Remove from open list to closed 
        
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            print('Path is found')
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares

            # Expansion
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Check boundaries
            if node_position[0] > (len(graph) - 1) or node_position[0] < 0 or node_position[1] > (len(graph[len(graph)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if graph[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node with its parent
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            skip = False
            for closed_child in closed_list:
                if child == closed_child:
                    skip= True
            if skip == True:
                continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            #Euclidean distance
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h
            
            skip = False
            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    skip= True
            if skip == True:
                continue

            # Add the child to the open list
            open_list.append(child)
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
path = A_star(obstacle_graph, start, end)

obstacle_graph[0][4]

astar = []

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


print(astar)