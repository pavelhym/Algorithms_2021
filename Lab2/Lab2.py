import numpy as np
from time import time
import matplotlib.pyplot as plt
import decimal
import pandas as pd



eps = 0.001
#1
#e = 0.001
#bruteforce
#1) f(x) = x**3 x[0,1]
#2) f(x) = |x-0.2| x[0,1]
#3) f(x) = x sin(1/x) x[0.01,1]


#1)
n = int((1-0)/eps)
f_x_hat = []
f_x_min = 1**3
x_min = 1
f_calc = 0
iter = 0
for k in range(0,n+1):
    iter += 1
    x_k = 0+k*(1-0)/n
    kth_value = x_k**3
    f_calc += 1
    f_x_hat.append(kth_value)
    if kth_value < f_x_min:
        f_x_min = kth_value
        x_min = x_k

print("Number of iterations brute force 1st = " + str(iter))
print("Number of f calc brute force 1st = " + str(f_calc))

graph_first = []
scale1 = []
for i in range(0,1000):
    graph_first.append((i/1000)**3)
    scale1.append(i/1000)


plt.plot(scale1,graph_first,label='function')
plt.plot(x_min,f_x_min,'ro',label = "bruteforce") 
plt.title("Brute force first")
plt.legend()
plt.show()

#2)

n2 = int((1-0)/eps)
f_x_hat2 = []
f_x_min2 = abs(1-0.2)
x_min2 = 1
f_calc2 = 0
iter2 = 0
for k in range(0,n2+1):
    iter2 +=1
    x_k = 0+k*(1-0)/n
    kth_value = abs(x_k-0.2)
    f_x_hat2.append(kth_value)
    f_calc2 += 1
    if kth_value < f_x_min2:
        f_x_min2 = kth_value
        x_min2 = x_k

print("Number of iterations brute force 2nd = " + str(iter2))
print("Number of f calc brute force 2nd = " + str(f_calc2))


graph_second = []
scale2 = []
for i in range(0,1000):
    graph_second.append(abs(i/1000-0.2))
    scale2.append(i/1000)

plt.plot(scale2,graph_second,label='function')
plt.plot(x_min2,f_x_min2,'ro',label = "bruteforce") 
plt.title("Brute force second")
plt.legend()
plt.show()

#3)



n3 = int((1-0.01)/eps)
f_x_hat3 = []
f_x_min3 = 1*np.sin(1/1)
x_min3 = 1
f_calc3 = 0
iter3 = 0
for k_big in range(1,(n3+1)*100):
    k=k_big/100
    iter3 +=1
    x_k = 0+k*(1-0)/n
    kth_value = x_k*np.sin(1/x_k)
    f_x_hat3.append(kth_value)
    f_calc3 += 1
    if kth_value < f_x_min3:
        f_x_min3 = kth_value
        x_min3 = x_k

print("Number of iterations brute force 3rd = " + str(iter3))
print("Number of f calc brute force 3rd = " + str(f_calc3))


graph_third = []
scale3 = []
for i in range(10,1000):
    i = i/1000
    graph_third.append(i*np.sin(1/i))
    scale3.append(i)

plt.plot(scale3,graph_third,label='function')
plt.plot(x_min3,f_x_min3,'ro',label = "bruteforce") 
plt.title("Brute force third")
plt.legend()
plt.show()



#dichotomy

def f1(x):
    return x**3

def f2(x):
    return abs(x-0.2)

def f3(x):
    return x*np.sin(1/x)

def dichotomy(fx,a,b,eps):
    a_0 = a
    b_0 = b
    x_1 = (a_0 + b_0 - eps/20)/2
    x_2 = (a_0 + b_0 + eps/20)/2
    f_calc_dic1 = 0
    iter_dic1 = 1
    a_1 = a
    b_1 = b
    while abs(a_1 - b_1)>= eps:
        if fx(x_1) <= fx(x_2):
            a_1 = a_1
            b_1 = x_2
        else:
            a_1 = x_1 
            b_1 = b_1

        f_calc_dic1 += 2
        iter_dic1 +=1
        x_1 = (a_1 + b_1 - eps/20)/2
        x_2 = (a_1 + b_1 + eps/20)/2
        



    x_min_dic1 = (b_1 + a_1)/2
    f_x_min_dic1 = fx(x_min_dic1)
    return x_min_dic1, f_x_min_dic1, iter_dic1, f_calc_dic1


print("Number of iterations Dichotomy 1st = " + str(dichotomy(f1,0,1,eps)[2]))
print("Number of f calc Dichotomy 1st = " + str(dichotomy(f1,0,1,eps)[3]))

plt.plot(scale1,graph_first,label='function')
plt.plot(dichotomy(f1,0,1,eps)[0],dichotomy(f1,0,1,eps)[1],'ro',label = "Dichotomy") 
plt.title("Dichotomy first")
plt.legend()
plt.show()



print("Number of iterations Dichotomy 2nd = " + str(dichotomy(f2,0,1,eps)[2]))
print("Number of f calc Dichotomy 2nd = " + str(dichotomy(f2,0,1,eps)[3]))

plt.plot(scale2,graph_second,label='function')
plt.plot(dichotomy(f2,0,1,eps)[0],dichotomy(f2,0,1,eps)[1],'ro',label = "Dichotomy") 
plt.title("Dichotomy second")
plt.legend()
plt.show()


print("Number of iterations Dichotomy 3rd = " + str(dichotomy(f3,0,1,eps)[2]))
print("Number of f calc Dichotomy 3rd = " + str(dichotomy(f3,0,1,eps)[3]))


plt.plot(scale3,graph_third,label='function')
plt.plot(dichotomy(f3,0.01,1,eps)[0],dichotomy(f3,0.01,1,eps)[1],'ro',label = "Dichotomy") 
plt.title("Dichotomy third")
plt.legend()
plt.show()


#Golden section

def golden_section(fx,a,b,eps):
    a_0 = a
    b_0 = b
    x_1 = a_0 + (3-np.sqrt(5))/2 * (b_0-a_0)
    x_2 = b_0 + (np.sqrt(5)-3)/2 * (b_0-a_0)
    fx_1 = fx(x_1)
    fx_2 = fx(x_2)
    f_calc = 2
    iter = 1
    while abs(a_0 - b_0)>= eps:
        iter += 1
        if fx_1 <= fx_2:
            a_0 = a_0
            b_0 = x_2
            x_2 = x_1
            x_1 = a_0 + (3-np.sqrt(5))/2 * (b_0-a_0)
            fx_2 = fx_1
            fx_1 = fx(x_1)
            f_calc += 1
        else:
            a_0 = x_1
            b_0 = b_0
            x_1 = x_2
            x_2 = b_0 + (np.sqrt(5)-3)/2 * (b_0-a_0)
            fx_1=fx_2
            fx_2 = fx(x_2)
            f_calc += 1
        #print(str(a_0)+ " - " + str(b_0))

    x_min_gold = (b_0 + a_0)/2
    f_x_min_gold = fx(x_min_gold)
    return x_min_gold, f_x_min_gold, iter, f_calc


print("Number of iterations Golden section 1st = " + str(golden_section(f1,0,1,eps)[2]))
print("Number of f calc Golden Section 1st = " + str(golden_section(f1,0,1,eps)[3]))

plt.plot(scale1,graph_first,label='function')
plt.plot(golden_section(f1,0,1,eps)[0],golden_section(f1,0,1,eps)[1],'ro',label = "Golden Section") 
plt.title("Golden Section first")
plt.legend()
plt.show()



print("Number of iterations Golden section 2nd = " + str(golden_section(f2,0,1,eps)[2]))
print("Number of f calc Golden section 2nd = " + str(golden_section(f2,0,1,eps)[3]))

plt.plot(scale2,graph_second,label='function')
plt.plot(golden_section(f2,0,1,eps)[0],golden_section(f2,0,1,eps)[1],'ro',label = "Golden Section") 
plt.title("Golden Section second")
plt.legend()
plt.show()


print("Number of iterations Golden Section 3rd = " + str(golden_section(f3,0,1,eps)[2]))
print("Number of f calc Golden Section 3rd = " + str(golden_section(f3,0,1,eps)[3]))


plt.plot(scale3,graph_third,label='function')
plt.plot(golden_section(f3,0.01,1,eps)[0],golden_section(f3,0.01,1,eps)[1],'ro',label = "Golden Section") 
plt.title("Golden Section third")
plt.legend()
plt.show()