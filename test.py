import numpy as np
from time import time
import matplotlib.pyplot as plt
import decimal
import pandas as pd

from timeit import Timer
import functools



def product(A,B):
    shape = len(A)
    result = np.zeros((shape,shape))
    for i in range(len(A)):

        # iterating by column by B
        for j in range(len(B[0])):
        
            # iterating by rows of B
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]


matr = []

for n in range(1,500,10):

    print(n)
    A = np.random.random((n, n))
    B = np.random.random((n, n))
    result = np.zeros((n,n))
    
    
    t = Timer(functools.partial(product,A,B))  
    time_cool = t.timeit(5)/5
    matr.append(time_cool)

n_num = []
for n in range(1,500,10):
    n_num.append(n)



final_df = pd.DataFrame()
final_df['n'] = n_num
final_df['time'] = matr

#final_df.to_csv("matrix_Timer.csv")

df=pd.read_csv('matrix_Timer.csv')




plt.plot(df['time'], label ='const')

#theoretical times




A = np.random.random((3, 3))
B = np.random.random((3, 3))


