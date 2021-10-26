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

for n in range(1,501,10):

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




#O(n**3)
theoretical_product = []
for n in range(1,500,10):
    theoretical_product.append(df['time'].tolist()[-1]/(500**3) *n**3 )
    #theoretical_quick_sort.append(df['quick_sort'].tolist()[-1]/2000**2 * n**2 )
plt.plot(df['time'], label ='matr_prod')
plt.plot(theoretical_product, label ='matr_prod_theor')
plt.legend()
plt.show()



