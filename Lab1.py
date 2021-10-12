import numpy as np
from time import time
import matplotlib.pyplot as plt
import decimal
import pandas as pd

from timeit import Timer
import functools


def constant(x):
    return 228

def sum(x):
    return np.sum(x)

def prod(x):
    return np.prod(x)

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
            break  # Stop iteration if the collection is sorted.
    return collection



def quick_sort(collection: list) -> list:
    if len(collection) < 2:
        return collection
    pivot = collection.pop()  # Use the last element as the first pivot
    greater: list[int] = []  # All elements greater than pivot
    lesser: list[int] = []  # All elements less than or equal to pivot
    for element in collection:
        (greater if element > pivot else lesser).append(element)
    return quick_sort(lesser) + [pivot] + quick_sort(greater)



def binary_search(lst, item, start, end):
    if start == end:
        return start if lst[start] > item else start + 1
    if start > end:
        return start

    mid = (start + end) // 2
    if lst[mid] < item:
        return binary_search(lst, item, mid + 1, end)
    elif lst[mid] > item:
        return binary_search(lst, item, start, mid - 1)
    else:
        return mid


def insertion_sort(lst):
    length = len(lst)

    for index in range(1, length):
        value = lst[index]
        pos = binary_search(lst, value, 0, index - 1)
        lst = lst[:pos] + [value] + lst[pos:index] + lst[index + 1 :]

    return lst


def merge(left, right):
    if not left:
        return right

    if not right:
        return left

    if left[0] < right[0]:
        return [left[0]] + merge(left[1:], right)

    return [right[0]] + merge(left, right[1:])


def tim_sort(lst):

    length = len(lst)
    runs, sorted_runs = [], []
    new_run = [lst[0]]
    sorted_array = []
    i = 1
    while i < length:
        if lst[i] < lst[i - 1]:
            runs.append(new_run)
            new_run = [lst[i]]
        else:
            new_run.append(lst[i])
        i += 1
    runs.append(new_run)

    for run in runs:
        sorted_runs.append(insertion_sort(run))
    for run in sorted_runs:
        sorted_array = merge(sorted_array, run)

    return sorted_array



n_num=[]
const1 = []
sum2 = []
prod3 = []
polin4 = []
polin_horn4 = []
bub_sort5 = []
quick_sort6 = []
timsort7 = []


for n in range(1,2000):
    #print(n)
    vec = np.random.rand(n).tolist()
    n_num.append(n)


    #const
    temp = []
    for i in range(0,5):
        t0 = time()
        constant(vec)
        t1 = time()
        temp.append(t1-t0)
    const1.append(np.mean(temp))

    #sum
    temp = []
    for i in range(0,5):
        t0 = time()
        sum(vec)
        t1 = time()
        temp.append(t1-t0)
    sum2.append(np.mean(temp))

    #prod
    temp = []
    for i in range(0,5):
        t0 = time()
        prod(vec)
        t1 = time()
        temp.append(t1-t0)
    prod3.append(np.mean(temp))

    #polin
    temp = []
    for i in range(0,5):
        t0 = time()
        poly(vec, 1.5)
        t1 = time()
        temp.append(t1-t0)
    polin4.append(np.mean(temp))

    temp = []
    for i in range(0,5):
        t0 = time()
        poly_horner(vec, 1.5)
        t1 = time()
        temp.append(t1-t0)
    polin_horn4.append(np.mean(temp))


    #bubble sort
    temp = []
    for i in range(0,5):
        t0 = time()
        bubble_sort(vec)
        t1 = time()
        temp.append(t1-t0)
    bub_sort5.append(np.mean(temp))

    #Quick Sort
    temp = []
    for i in range(0,5):
        t0 = time()
        quick_sort(vec)
        t1 = time()
        temp.append(t1-t0)
    quick_sort6.append(np.mean(temp))

    #timsort
    temp = []
    for i in range(0,5):
        t0 = time()
        tim_sort(vec)
        t1 = time()
        temp.append(t1-t0)
    timsort7.append(np.mean(temp))


const1 = []
sum2 = []
prod3 = []
polin4 = []
polin_horn4 = []
bub_sort5 = []
quick_sort6 = []
timsort7 = []

plt.plot(const1, label ='const')
plt.plot(sum2, label ='sum')
plt.plot(prod3,label = 'prod')
plt.plot(polin4,label = 'polin')
plt.plot(polin_horn4,label = 'polin_horn')
plt.plot(bub_sort5,label = 'bub_sort')
plt.plot(quick_sort6,label = 'quick_sort')
plt.plot(timsort7,label = 'timsort')

plt.plot(polin4,label = 'polin')

plt.legend()
plt.show()

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

#final_df.to_csv("sorting_values.csv")

df=pd.read_csv('sorting_values.csv')

plt.plot(df['polin'],label = 'polin')
plt.plot(df['sum'], label ='sum')

plt.legend()
plt.show()


n_num=[]
sum1 = []
sum2 = []
sum3 = []

for n in range(1,2000):
    #print(n)
    vec = np.random.rand(n).tolist()
    n_num.append(n)


    #sum
    temp = []
    for i in range(0,5):
        t0 = time()
        sum(vec)
        t1 = time()
        temp.append(t1-t0)
    sum1.append(np.mean(temp))

    temp = []
    for i in range(0,5):
        t = Timer(functools.partial(sum,vec))  
        time_cool = t.timeit(1)
        temp.append(time_cool)
    sum2.append(np.mean(temp))

    t = Timer(functools.partial(sum,vec))  
    time_cool = t.timeit(5)/5
    sum3.append(time_cool)











n_num=[]
const1 = []
sum2 = []
prod3 = []
polin4 = []
polin_horn4 = []
bub_sort5 = []
quick_sort6 = []
timsort7 = []


for n in range(1,2000):
    #print(n)
    vec = np.random.rand(n).tolist()
    n_num.append(n)



    #const
    t = Timer(functools.partial(constant,vec))  
    time_eval = t.timeit(5)/5
    const1.append(time_eval)

    #sum
    t = Timer(functools.partial(sum,vec))  
    time_eval = t.timeit(5)/5
    sum2.append(time_eval)

    #prod

    t = Timer(functools.partial(prod,vec))  
    time_eval = t.timeit(5)/5
    prod3.append(time_eval)

    #polin
    t = Timer(functools.partial(poly,vec,1.5))  
    time_eval = t.timeit(5)/5
    polin4.append(time_eval)

    t = Timer(functools.partial(poly_horner,vec,1.5))  
    time_eval = t.timeit(5)/5
    polin_horn4.append(time_eval)


    #bubble sort
    t = Timer(functools.partial(bubble_sort,vec))  
    time_eval = t.timeit(5)/5
    bub_sort5.append(time_eval)

    #Quick Sort
    t = Timer(functools.partial(quick_sort,vec))  
    time_eval = t.timeit(5)/5
    quick_sort6.append(time_eval)

    #timsort
    t = Timer(functools.partial(tim_sort,vec))  
    time_eval = t.timeit(5)/5
    timsort7.append(time_eval)



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

#final_df.to_csv("sorting_values_Timer.csv")



plt.plot(const1, label ='const')
plt.plot(sum2, label ='sum')
plt.plot(prod3,label = 'prod')
plt.plot(polin4,label = 'polin')
plt.plot(polin_horn4,label = 'polin_horn')
plt.plot(df['polin_horn'],label = 'polin_horn')

plt.plot(bub_sort5,label = 'bub_sort')
plt.plot(quick_sort6,label = 'quick_sort')
plt.plot(timsort7,label = 'timsort')

plt.plot(polin4,label = 'polin')

plt.legend()
plt.show()