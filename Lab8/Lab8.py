import numpy as np
from time import time
import matplotlib.pyplot as plt
import decimal
import pandas as pd
from collections import defaultdict
import copy

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import networkx
from networkx import Graph


#Functions For creating random weighted connected graph (which fits MST problem)
#############
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
            if y[i] > 0:
                res[start].append(i)
        start += 1
    return res


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
            #print(group)
            visited = visited + group
            components.append(group)
            group = []
            grop_n +=1
    components.pop(0)
    return grop_n, components


def random_connected_weighted_graph(V,E):
    if V - E > 1:
        return print("Ureal")
    adj_matrix =  random_adjacency_matrix_weighted(V,E)
    adj_list =  convert_to_adjacency(adj_matrix)
    while find_components(adj_list)[0] > 1:
        adj_matrix =  random_adjacency_matrix_weighted(V,E)
        adj_list =  convert_to_adjacency(adj_matrix)
    return adj_matrix



def graph_matrix(adj_matrix):
    adj_matrix_unweighted =  copy.deepcopy(adj_matrix)
    adj_matrix_unweighted[adj_matrix_unweighted>0] = 1
    adj_sparse = sp.sparse.coo_matrix(adj_matrix_unweighted, dtype=np.int8)
    labels = range(0,len(adj_matrix))
    DF_adj = pd.DataFrame(adj_sparse.toarray(),index=labels,columns=labels)
    #print(DF_adj)



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

###########





def convert_to_adjacency_weighted(adj_matrix):
    graph = defaultdict(dict)
    edges = set()

    for i, v in enumerate(adj_matrix, 0):
        for j, u in enumerate(v, 0):
            if u != 0 :
                edges.add(frozenset([i, j]))
                graph[i].update({j: u})
                #print(edges)
    return graph





def convert_to_edges(adj_matrix):
    edgelist = []
    for i, v in enumerate(adj_matrix, 0):
        for j, u in enumerate(v, 0):
            if u != 0 :
                if [j,i,u] not in edgelist:
                    edgelist.append([i,j,u])

    return edgelist



#Kruskal algorithm

def join(X, Y, r, p):
    a = get(X,r, p); b = get(Y, r, p)
    if r[b] > r[a]:
        a, b = b, a
    elif r[a] == r[b]:
        r[a] += 1
    p[b] = a
    
def get(v, r, p):
    if p[v] == v:
        return v 
    else:
        p[v] = get(p[v], r, p)
        return p[v]

def Kruskal(edge_list, vert, edges):

    n, m = vert, edges
    E = edge_list
    p = [i for i in range(n)]
    r = [0] * n


    E.sort(key = lambda x: x[2])

    ans = 0
    connections = []
    n_ver = 0
    while n_ver < vert - 1:
        for e in E:
            if get(e[0], r, p) != get(e[1], r, p):
                join(e[0], e[1], r, p)
                connections.append(e)
                ans += e[2]
                n_ver += 1
    return ans, connections 




def graph_with_path(edge_list, path):
    
    G = nx.Graph()
    for edge in edge_list:
        edge_inverse = [0]*3
        edge_inverse[0], edge_inverse[1], edge_inverse[2] = edge[1], edge[0], edge[2]
        if (edge in path) or (edge_inverse in path):
            G.add_edge(edge[0],edge[1],color='r',weight=edge[2])
        else:
            G.add_edge(edge[0],edge[1],color='b',weight= edge[2])
    
    
    labels = nx.get_edge_attributes(G,'weight')
    
    pos = nx.circular_layout(G)
    
    edges = G.edges()
    colors = [G[u][v]['color'] for u,v in edges]
    weights = [G[u][v]['weight'] for u,v in edges]
    nx.draw(G,pos,edge_color=colors,with_labels=True)
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)





#Another Kruskul implementation


 
 

#Prim for matrix O(V^2)
import sys

def minKey(key, mstSet):

    # Initialize min value
    min = sys.maxsize

    for v in range(len(adj_matrix)):
        if key[v] < min and mstSet[v] == False:
            min = key[v]
            min_index = v

    return min_index



def primMST(adj_matrix):

    # Key values used to pick minimum weight edge in cut
    key = [sys.maxsize] * len(adj_matrix)
    parent = [None] * len(adj_matrix) # Array to store constructed MST
    # Make key 0 so that this vertex is picked as first vertex
    key[0] = 0
    mstSet = [False] * len(adj_matrix)

    parent[0] = -1 # First node is always the root of

    for cout in range(len(adj_matrix)):

        # Pick the minimum distance vertex from
        # the set of vertices not yet processed.
        # u is always equal to src in first iteration
        u = minKey(key, mstSet)

        # Put the minimum distance vertex in
        # the shortest path tree
        mstSet[u] = True

        # Update dist value of the adjacent vertices
        # of the picked vertex only if the current
        # distance is greater than new distance and
        # the vertex in not in the shortest path tree
        for v in range(len(adj_matrix)):

            # graph[u][v] is non zero only for adjacent vertices of m
            # mstSet[v] is false for vertices not yet included in MST
            # Update the key only if graph[u][v] is smaller than key[v]
            if adj_matrix[u][v] > 0 and mstSet[v] == False and key[v] > adj_matrix[u][v]:
                    key[v] = adj_matrix[u][v]
                    parent[v] = u


    
    return parent

    edgelist = []
    for i in range(1,len(adj_matrix)):
        edgelist.append([i,parent[i],adj_matrix[i,parent[i]] ])


#conver to list of edges
def Prim_matrix_to_edges(parent):
    edgelist = []
    for i in range(1,len(parent)):
        edgelist.append([i,parent[i],adj_matrix[i,parent[i]] ])
    return edgelist





#Prim for adj list



from collections import defaultdict
import heapq



def prim_list_heap(graph, starting_vertex):
    mst = defaultdict(set)
    visited = set([starting_vertex])
    edges = [(cost, starting_vertex, to) for to, cost in graph[starting_vertex].items()]
    heapq.heapify(edges)

    while edges:
        cost, frm, to = heapq.heappop(edges)
        if to not in visited:
            visited.add(to)
            mst[frm].add(to)
            for to_next, cost in graph[to].items():
                if to_next not in visited:
                    heapq.heappush(edges, (cost, to, to_next))

    return mst



def Prim_list_to_edges(result_Prima_span):
    edgelist = []
    for i in result_Prima_span.keys():
        start =  i
        for end in result_Prima_span[i]:
            dist = adj_matrix[start][end]
            if ([start,end, dist] not in edgelist) or ([end,start, dist] not in edgelist):
                 edgelist.append([start,end, dist])
    return edgelist




from FibHeap import FibHeapForPrims


def prim_list_fheap(adj_list):
	precursor = [None] * len(adj_list)
	visited = set()

	src = 0
	heap = FibHeapForPrims(len(adj_list))

	for _ in range(len(adj_list)):
		u = heap.extract_min()
		#print("After extraction: \n"); heap.print_heap()
		visited.add(u)

		for v, cur_key in adj_list[u].items():
			if v not in visited and cur_key < heap.fetch_key(v):
				heap.decrease_key(v, cur_key) 
				#print(f"After decrease_key on {v}: \n"); heap.print_heap()
				precursor[v] = u

	return precursor





#Results
#Creating random graph

adj_matrix =  random_connected_weighted_graph(10,20)
adj_list =  convert_to_adjacency_weighted(adj_matrix)
edge_list =  convert_to_edges(adj_matrix)


#plot with weights
graph_with_path(edge_list, [])
plt.title("Random graph")
plt.savefig('Plots\graph.png')
plt.show()
#Graphs for representation


#Kruskal
result_Kruskal = Kruskal(edge_list, 10, 20)
graph_with_path(edge_list, result_Kruskal[1])
plt.title("Kruskal")
plt.savefig('Plots\graph_kruskal.png')
plt.show()


#Prim matrix
prim_matrix_result =  primMST(adj_matrix)
graph_with_path(edge_list, Prim_matrix_to_edges(prim_matrix_result))
plt.title("Prim's matrix")
plt.savefig('Plots\graph_prim_matr.png')
plt.show()

#Prim list
prim_list_result =  prim_list_heap(adj_list, 0)
graph_with_path(edge_list, Prim_list_to_edges(prim_list_result))
plt.title("Prim's list")
plt.savefig('Plots\graph_prim_list.png')
plt.show()


#Prim list Fibonacci heap

prim_list_fheap_result =  prim_list_fheap(adj_list)
graph_with_path(edge_list, Prim_matrix_to_edges(prim_list_fheap_result))
plt.title("Prim's list Fibonacci")
plt.savefig('Plots\graph_prim_list_fib.png')
plt.show()

#Estimation of time complexity with growing number of V and E


from timeit import Timer
import functools



V_list = []
E_list = []
krusk = []
prim_matr = []
prim_list = []
prim_list_fib = []




for V in range(10,501,10):
    #E = int(V*(V-1)/2 -10)
    E = V*3
    V_list.append(V)
    E_list.append(E)
    krusk_temp = []
    prim_matr_temp = []
    prim_list_temp = []
    prim_list_fib_temp = []
    for i in range(0,5):
        adj_matrix =  random_connected_weighted_graph(V,E)
        adj_list =  convert_to_adjacency_weighted(adj_matrix)
        edge_list =  convert_to_edges(adj_matrix)

        t = Timer(functools.partial(Kruskal,edge_list, V, E))  
        time_eval = t.timeit(1)
        krusk_temp.append(time_eval)


        t = Timer(functools.partial(primMST,adj_matrix))  
        time_eval = t.timeit(1)
        prim_matr_temp.append(time_eval)

        t = Timer(functools.partial(prim_list_heap,adj_list, 0))  
        time_eval = t.timeit(1)
        prim_list_temp.append(time_eval)


        t = Timer(functools.partial(prim_list_fheap,adj_list))  
        time_eval = t.timeit(1)
        prim_list_fib_temp.append(time_eval)
    
    krusk.append(np.mean(krusk_temp))
    prim_matr.append(np.mean(prim_matr_temp))
    prim_list.append(np.mean(prim_list_temp))
    prim_list_fib.append(np.mean(prim_list_fib_temp))



plt.plot(V_list, krusk, label = "kurs")
plt.plot(V_list, prim_matr, label = 'prim matr')
plt.plot(V_list, prim_list, label = 'prim list')
plt.plot(V_list, prim_list_fib, label = 'prim Fibbonaci')
plt.legend()
plt.title("Comparison on graph with small number of edges")
plt.savefig('Plots\small_comparison.png')

krusk_theor = []
for V in range(10,501,10):
    #E = int(V*(V-1)/2 -10)
    E = V*3
    krusk_theor.append((krusk[-1]/(1500*np.log(500)))*E*np.log(V))

plt.plot(V_list, krusk, label = "Kurskal")
plt.plot(V_list, krusk_theor,label = "O(Elog(V))") 
plt.legend()
plt.title("Kurskal theoretical comparison")
plt.savefig('Plots\krusk_theor.png')


prim_matr_theor = []
for V in range(10,501,10):
    #E = int(V*(V-1)/2 -10)
    E = V*3
    prim_matr_theor.append((prim_matr[-1]/(500**2)*V**2))

plt.plot(V_list, prim_matr, label = "Prim's matrix")
plt.plot(V_list, prim_matr_theor,label = "O(V^2)")
plt.legend()
plt.title("Prim's matrix theoretical comparison") 
plt.savefig('Plots\prim_matr_theor.png')


prim_list_theor = []
for V in range(10,501,10):
    #E = int(V*(V-1)/2 -10)
    E = V*3
    prim_list_theor.append((prim_list[-1]/(1500*np.log(500)))*E*np.log(V))

plt.plot(V_list, prim_list, label = "Prim's list")
plt.plot(V_list, prim_list_theor,label = "O(Elog(V))")
plt.legend()
plt.title("Prim's list theoretical comparison") 
plt.savefig('Plots\prim_list_theor.png')

#O(E+V log(V))
prim_list_fib_theor = []
for V in range(10,501,10):
    #E = int(V*(V-1)/2 -10)
    E = V*3
    prim_list_fib_theor.append((prim_list_fib[-1]/(1500+ 500*np.log(500)))*(E+V*np.log(V)))

plt.plot(V_list, prim_list_fib, label = "Prim's list fib")
plt.plot(V_list, prim_list_fib_theor,label = "O(E + Vlog(V))")
plt.legend()
plt.title("Prim's list with fibonacci theoretical comparison") 
plt.savefig('Plots\prim_list_fib_theor.png')



#DENSE GRAPH



V_list_dense = []
E_list_dense = []
krusk_dense = []
prim_matr_dense = []
prim_list_dense = []
prim_list_fib_dense = []



for V in range(10,260,10):
    E = int(V*(V-1)/2)
    #E = V*3
    V_list_dense.append(V)
    E_list_dense.append(E)
    krusk_temp = []
    prim_matr_temp = []
    prim_list_temp = []
    prim_list_fib_temp = []
    for i in range(0,5):
        adj_matrix =  random_connected_weighted_graph(V,E)
        adj_list =  convert_to_adjacency_weighted(adj_matrix)
        edge_list =  convert_to_edges(adj_matrix)

        t = Timer(functools.partial(Kruskal,edge_list, V, E))  
        time_eval = t.timeit(1)
        krusk_temp.append(time_eval)


        t = Timer(functools.partial(primMST,adj_matrix))  
        time_eval = t.timeit(1)
        prim_matr_temp.append(time_eval)

        t = Timer(functools.partial(prim_list_heap,adj_list, 0))  
        time_eval = t.timeit(1)
        prim_list_temp.append(time_eval)


        t = Timer(functools.partial(prim_list_fheap,adj_list))  
        time_eval = t.timeit(1)
        prim_list_fib_temp.append(time_eval)
    
    krusk_dense.append(np.mean(krusk_temp))
    prim_matr_dense.append(np.mean(prim_matr_temp))
    prim_list_dense.append(np.mean(prim_list_temp))
    prim_list_fib_dense.append(np.mean(prim_list_fib_temp))


plt.plot(V_list_dense, krusk_dense, label = "kurs")
plt.plot(V_list_dense, prim_matr_dense, label = 'prim matr')
plt.plot(V_list_dense, prim_list_dense, label = 'prim list')
plt.plot(V_list_dense, prim_list_fib_dense, label = 'prim Fibbonaci')
plt.legend()
plt.title("Comparison on graph with large number of edges")
plt.savefig('Plots\cbig_comparison.png')


krusk_theor_dense = []
for V in range(10,260,10):
    E = int(V*(V-1)/2)
    krusk_theor_dense.append((krusk_dense[-1]/(31125*np.log(250)))*E*np.log(V))

plt.plot(V_list_dense, krusk_dense, label = "Kurskal")
plt.plot(V_list_dense, krusk_theor_dense,label = "O(Elog(V))") 
plt.legend()
plt.title("Kurskal dense theoretical comparison")
plt.savefig('Plots\krusk_theor_big.png')






prim_matr_theor_dense = []
for V in range(10,260,10):
    E = int(V*(V-1)/2)
    prim_matr_theor_dense.append((prim_matr_dense[-1]/(250**2)*V**2))

plt.plot(V_list_dense, prim_matr_dense, label = "Prim's matrix")
plt.plot(V_list_dense, prim_matr_theor_dense,label = "O(V^2)")
plt.legend()
plt.title("Prim's matrix dense theoretical comparison") 
plt.savefig('Plots\prim_matr_theor_big.png')


prim_list_theor_dense = []
for V in range(10,260,10):
    E = int(V*(V-1)/2)
    prim_list_theor_dense.append((prim_list_dense[-1]/(31125*np.log(250)))*E*np.log(V))

plt.plot(V_list_dense, prim_list_dense, label = "Prim's list")
plt.plot(V_list_dense, prim_list_theor_dense,label = "O(Elog(V))")
plt.legend()
plt.title("Prim's list dense theoretical comparison") 
plt.savefig('Plots\prim_list_theor_big.png')


#O(E+V log(V))
prim_list_fib_theor_dense = []
for V in range(10,260,10):
    E = int(V*(V-1)/2)
    prim_list_fib_theor_dense.append((prim_list_fib_dense[-1]/(31125+ 250*np.log(250)))*(E+V*np.log(V)))

plt.plot(V_list_dense, prim_list_fib_dense, label = "Prim's list fib")
plt.plot(V_list_dense, prim_list_fib_theor_dense,label = "O(E + Vlog(V))")
plt.legend()
plt.title("Prim's dense list with fibonacci theoretical comparison") 
plt.savefig('Plots\prim_list_fib_theor_big.png')
