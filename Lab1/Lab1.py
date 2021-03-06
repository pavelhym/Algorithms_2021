import numpy as np
from time import time
import matplotlib.pyplot as plt
import decimal
import pandas as pd

from timeit import Timer
import functools


def constant(x):
    return 228


def prod(myList) :
     
    # Multiply elements one by one
    result = 1
    for x in myList:
         result = result * x
    return result


def poly(A,x):
    p = 0
    for i in range(0,len(A)):
        p += decimal.Decimal(A[i])*(decimal.Decimal(x))**i
    return p



def poly_horner(A, x):
    p = A[-1]
    i = len(A) - 2
    while i >= 0:
        p = p * x + A[i]
        i -= 1
    return p


def bubble_sort(collection):
    length = len(collection)
    for i in range(length - 1):
        swapped = False
        for j in range(length - 1 - i):
            if collection[j] > collection[j + 1]:
                swapped = True
                collection[j], collection[j + 1] = collection[j + 1], collection[j]
        if not swapped:
            break  
    return collection


def quick_sort(array):
    

    less = []
    equal = []
    greater = []

    if len(array) > 1:
        pivot = array[0]
        for x in array:
            if x < pivot:
                less.append(x)
            elif x == pivot:
                equal.append(x)
            elif x > pivot:
                greater.append(x)
        
        return quick_sort(less)+equal+quick_sort(greater)  
    
    else:  
        return array


MINIMUM= 32
  
def find_minrun(n): 
  
    r = 0
    while n >= MINIMUM: 
        r |= n & 1
        n >>= 1
    return n + r 
  
def insertion_sort(array, left, right): 
    for i in range(left+1,right+1):
        element = array[i]
        j = i-1
        while element<array[j] and j>=left :
            array[j+1] = array[j]
            j -= 1
        array[j+1] = element
    return array
              
def merge(array, l, m, r): 
  
    array_length1= m - l + 1
    array_length2 = r - m 
    left = []
    right = []
    for i in range(0, array_length1): 
        left.append(array[l + i]) 
    for i in range(0, array_length2): 
        right.append(array[m + 1 + i]) 
  
    i=0
    j=0
    k=l
   
    while j < array_length2 and  i < array_length1: 
        if left[i] <= right[j]: 
            array[k] = left[i] 
            i += 1
  
        else: 
            array[k] = right[j] 
            j += 1
  
        k += 1
  
    while i < array_length1: 
        array[k] = left[i] 
        k += 1
        i += 1
  
    while j < array_length2: 
        array[k] = right[j] 
        k += 1
        j += 1
  
def tim_sort(array): 
    n = len(array) 
    minrun = find_minrun(n) 
  
    for start in range(0, n, minrun): 
        end = min(start + minrun - 1, n - 1) 
        insertion_sort(array, start, end) 
   
    size = minrun 
    while size < n: 
  
        for left in range(0, n, 2 * size): 
  
            mid = min(n - 1, left + size - 1) 
            right = min((left + 2 * size - 1), (n - 1)) 
            merge(array, left, mid, right) 
  
        size = 2 * size 




n_num=[]
const1 = []
sum2 = []
prod3 = []
polin4 = []
polin_horn4 = []
bub_sort5 = []
quick_sort6 = []
timsort7 = []

for n in range(1,2001):
    #sum
    temp = []
    for i in range(0,5):
        vec = np.random.rand(n).tolist()
        t = Timer(functools.partial(sum,vec))  
        time_eval = t.timeit(1)
        temp.append(time_eval)
    sum2.append(np.mean(temp))

    #prod
    temp = []
    for i in range(0,5):
        vec = np.random.rand(n).tolist()
        t = Timer(functools.partial(prod,vec))  
        time_eval = t.timeit(1)
        temp.append(time_eval)
    prod3.append(np.mean(temp))




for n in range(1,2001):
    print(n)
    #vec = np.random.rand(n).tolist()
    n_num.append(n)



    #const
    temp = []
    for i in range(0,5):
        vec = np.random.rand(n).tolist()
        t = Timer(functools.partial(constant,vec))  
        time_eval = t.timeit(1)
        temp.append(time_eval)
    const1.append(np.mean(temp))

    #sum
    temp = []
    for i in range(0,5):
        vec = np.random.rand(n).tolist()
        t = Timer(functools.partial(sum,vec))  
        time_eval = t.timeit(1)
        temp.append(time_eval)
    sum2.append(np.mean(temp))

    #prod
    temp = []
    for i in range(0,5):
        vec = np.random.rand(n).tolist()
        t = Timer(functools.partial(prod,vec))  
        time_eval = t.timeit(1)
        temp.append(time_eval)
    prod3.append(np.mean(temp))

    #polin
    temp = []
    for i in range(0,5):
        vec = np.random.rand(n).tolist()
        t = Timer(functools.partial(poly,vec,1.5))  
        time_eval = t.timeit(1)
        temp.append(time_eval)
    polin4.append(np.mean(temp))

    temp = []
    for i in range(0,5):
        vec = np.random.rand(n).tolist()
        t = Timer(functools.partial(poly_horner,vec,1.5))  
        time_eval = t.timeit(1)
        temp.append(time_eval)
    polin_horn4.append(np.mean(temp))


    #bubble sort
    temp = []
    for i in range(0,5):
        vec = np.random.rand(n).tolist()
        t = Timer(functools.partial(bubble_sort,vec))  
        time_eval = t.timeit(1)
        temp.append(time_eval)
    bub_sort5.append(np.mean(temp))

    #Quick Sort
    temp = []
    for i in range(0,5):
        vec = np.random.rand(n).tolist()
        t = Timer(functools.partial(quick_sort,vec))  
        time_eval = t.timeit(1)
        temp.append(time_eval)
    quick_sort6.append(np.mean(temp))

    #timsort
    temp = []
    for i in range(0,5):
        vec = np.random.rand(n).tolist()
        t = Timer(functools.partial(tim_sort,vec))  
        time_eval = t.timeit(1)
        temp.append(time_eval)
    timsort7.append(np.mean(temp))



final_df = pd.DataFrame()
final_df['n'] = n_num
final_df['const'] = const1
final_df['sum'] = sum2
final_df['prod'] = prod3
final_df['polin'] = polin4
final_df['polin_horn'] = polin_horn4
final_df['bub_sort'] = bub_sort5
final_df['quick_sort'] = quick_sort6
final_df['timsort'] = timsort7

#final_df.to_csv("final_sorting_values_Timer.csv")

final_df = pd.read_csv("final_sorting_values_Timer.csv")
df=final_df



#GRAPHS

#Theoretical
#const - const

theoretical_const = []
for n in range(1,2001):
    theoretical_const.append(df['const'].tolist()[-1])

plt.plot(df['const'], label ='const')
plt.plot(theoretical_const, label ='const_theor')
plt.legend()
plt.title("Constant function O(1)")
plt.savefig('Plots\const.png')
plt.show()


#Sum

theoretical_sum = []
for n in range(1,2001):
    theoretical_sum.append(df['sum'].tolist()[-1]/2001 *n)

plt.plot(df['sum'], label ='sum')
plt.plot(theoretical_sum, label ='sum_theor')
plt.legend()
plt.title("Sum function O(n)")
plt.savefig('Plots\sum.png')
plt.show()


#prod 
#O(n)

theoretical_prod = []
for n in range(1,2001):
    theoretical_prod.append(df['prod'].tolist()[-1]/2001 *n )

plt.plot(df['prod'], label ='prod')
plt.plot(theoretical_prod, label ='prod_theor')
plt.legend()
plt.title("Production function O(n)")
plt.savefig('Plots\prod.png')
plt.show()


#polin
#mb O(n)

theoretical_polin = []
for n in range(1,2001):
    theoretical_polin.append(df['polin'].tolist()[-1]/2001 *n )

plt.plot(df['polin'], label ='polin')
plt.plot(theoretical_polin, label ='polin_theor')
plt.legend()
plt.title("Standard polynomial O(n)")
plt.savefig('Plots\poly.png')
plt.show()


#polin horn

theoretical_polin_horn = []
for n in range(1,2001):
    theoretical_polin_horn.append(df['polin_horn'].tolist()[-1]/2001 *n )

plt.plot(df['polin_horn'], label ='polin_horn')
plt.plot(theoretical_polin_horn, label ='polin_horn_theor')
plt.legend()
plt.title("Horner's polynomial O(n)")
plt.savefig('Plots\poly_horner.png')
plt.show()

#polins comparison
plt.plot(df['polin'], label ='polin')
plt.plot(df['polin_horn'], label ='polin_horn')
plt.title("Polynomial comparison")
plt.legend()
plt.savefig('Plots\poly_comp.png')
plt.show()


#bub_sort
#O(n**2)

theoretical_bub_sort = []
for n in range(1,2001):
    #theoretical_bub_sort.append(df['bub_sort'].tolist()[-1]/(1999*np.log(1999)) *n*np.log(n) )
    theoretical_bub_sort.append(df['bub_sort'].tolist()[-1]/2001**2 * n**2 )
plt.plot(df['bub_sort'], label ='bub_sort')
plt.plot(theoretical_bub_sort, label ='bub_sort_theor')
plt.legend()
plt.title("Bub sort O(n**2)")
plt.savefig('Plots\obub_sort.png')
plt.show()



#quick_sort
#O(nlogn)
theoretical_quick_sort = []
for n in range(1,2001):
    theoretical_quick_sort.append(df['quick_sort'].tolist()[-1]/(2000*np.log(2000)) *n*np.log(n) )
    #theoretical_quick_sort.append(df['quick_sort'].tolist()[-1]/2000**2 * n**2 )
plt.plot(df['quick_sort'], label ='quick_sort')
plt.plot(theoretical_quick_sort, label ='quick_sort_theor')
plt.legend()
plt.title("Quick sort O(n log(n))")
plt.savefig('Plots\quick_sort.png')
plt.show()



#timsort 


theoretical_timsort = []
for n in range(1,2001):
    theoretical_timsort.append(df['timsort'].tolist()[-2]/(2001*np.log(2001)) *n*np.log(n) )
    #theoretical_timsort.append(df['timsort'].tolist()[-1]/1999**2 * n**2 )
plt.plot(df['timsort'], label ='tim_sort')
plt.plot(theoretical_timsort, label ='tim_sort_theor')
plt.legend()
plt.title("Timsort average O(n log(n))")
plt.savefig('Plots\otim_sort.png')
plt.show()


#sort comparison
plt.plot(df['timsort'], label ='tim_sort')
plt.plot(df['quick_sort'], label ='quick_sort')
plt.plot(df['bub_sort'], label ='bub_sort')
plt.legend()
plt.title("Comparison of sorting")
plt.savefig('Plots\sort_comp.png')
plt.show()





##3

def product(A,B):
    shape = len(A)
    result = np.zeros((shape,shape))
    for i in range(len(A)):

        # iterating by column by B
        for j in range(len(B[0])):
        
            # iterating by rows of B
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]


A = np.random.random((3, 3))
type(A)


n_num = []
matr = []

for n in range(1,501,10):
    n_num.append(n)
    print(n)

    result = np.zeros((n,n))
    
    temp = []
    for i in range(0,5): 
        A = np.random.random((n, n))
        B = np.random.random((n, n))   
        t = Timer(functools.partial(product,A,B))  
        time_eval = t.timeit(1)
        temp.append(time_eval)
    matr.append(np.mean(temp))


final_df = pd.DataFrame()
final_df['n'] = n_num
final_df['time'] = matr

#final_df.to_csv("matrix_Timer.csv")

df_matrix=pd.read_csv('matrix_Timer.csv')

#theoretical times




#O(n**3)
theoretical_product = []
for n in range(1,500,10):
    theoretical_product.append(df_matrix['time'].tolist()[-1]/(500**3) *n**3 )
    #theoretical_quick_sort.append(df['quick_sort'].tolist()[-1]/2000**2 * n**2 )
plt.plot(df_matrix['n'],df_matrix['time'], label ='matr_prod')
plt.plot(df_matrix['n'],theoretical_product, label ='matr_prod_theor')
plt.legend()
plt.title("Matrix product O(n**3)")
plt.savefig('Plots\matrix.png')
plt.show()




  
  
  
  
  
  