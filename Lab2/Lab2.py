import numpy as np
from time import time
import matplotlib.pyplot as plt
import decimal
import pandas as pd

import random

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import copy



eps = 0.001

def f1(x):
    return x**3

def f2(x):
    return abs(x-0.2)

def f3(x):
    return x*np.sin(1/x)
#1
#e = 0.001
#bruteforce
#1) f(x) = x**3 x[0,1]
#2) f(x) = |x-0.2| x[0,1]
#3) f(x) = x sin(1/x) x[0.01,1]



def bruteforce(fx,a,b,eps):
    n = int((b-a)/eps)
    f_x_min = fx(a)
    x_min = a
    f_calc = 1
    iter = 1
    for k in range(int((a+1)*100),(n+1)*100,100):
        k = k/100
        iter += 1
        x_k = a+k*(b-a)/n
        kth_value = fx(x_k)
        f_calc += 1
        if kth_value < f_x_min:
            f_x_min = kth_value
            x_min = x_k 
    return   x_min, f_x_min, iter, f_calc




#dichotomy



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






#Results for 1_1

print("x_min brute force 1st = " + str(bruteforce(f1,0,1,eps)[0]))
print("f_min brute force 1st = " + str(bruteforce(f1,0,1,eps)[1]))
print("Number of iterations brute force 1st = " + str(bruteforce(f1,0,1,eps)[2]))
print("Number of f calc brute force 1st = " + str(bruteforce(f1,0,1,eps)[3]))

print("x_min Dichotomy 1st = " + str(dichotomy(f1,0,1,eps)[0]))
print("f_min  Dichotomy 1st = " + str(dichotomy(f1,0,1,eps)[1]))
print("Number of iterations Dichotomy 1st = " + str(dichotomy(f1,0,1,eps)[2]))
print("Number of f calc Dichotomy 1st = " + str(dichotomy(f1,0,1,eps)[3]))

print("x_min Golden section 1st = " + str(golden_section(f1,0,1,eps)[0]))
print("f_min Golden Section 1st = " + str(golden_section(f1,0,1,eps)[1]))
print("Number of iterations Golden section 1st = " + str(golden_section(f1,0,1,eps)[2]))
print("Number of f calc Golden Section 1st = " + str(golden_section(f1,0,1,eps)[3]))



graph_first = []
scale1 = []
for i in range(0,1000):
    graph_first.append((i/1000)**3)
    scale1.append(i/1000)


plt.plot(scale1[0:10],graph_first[0:10],label='x**3')
plt.plot(bruteforce(f1,0,1,eps)[0],bruteforce(f1,0,1,eps)[1],'ro',label = "bruteforce") 
plt.plot(dichotomy(f1,0,1,eps)[0],dichotomy(f1,0,1,eps)[1],'bo',label = "dichotomy") 
plt.plot(golden_section(f1,0,1,eps)[0],golden_section(f1,0,1,eps)[1],'go',label = "golden_section") 
plt.title("Algoritms comparison")
plt.legend()
plt.savefig('Plots\comp_1_1.png')
plt.show()


#Results for 1_2

print("x_min brute force 2nd = " + str(bruteforce(f2,0,1,eps)[0]))
print("f_min brute force 2nd = " + str(bruteforce(f2,0,1,eps)[1]))
print("Number of iterations brute force 2nd = " + str(bruteforce(f2,0,1,eps)[2]))
print("Number of f calc brute force 2nd = " + str(bruteforce(f2,0,1,eps)[3]))


print("x_min Dichotomy 2nd = " + str(dichotomy(f2,0,1,eps)[0]))
print("f_min Dichotomy 2nd = " + str(dichotomy(f2,0,1,eps)[1]))
print("Number of iterations Dichotomy 2nd = " + str(dichotomy(f2,0,1,eps)[2]))
print("Number of f calc Dichotomy 2nd = " + str(dichotomy(f2,0,1,eps)[3]))

print("x_min Golden section 2nd = " + str(golden_section(f2,0,1,eps)[0]))
print("f_min Golden section 2nd = " + str(golden_section(f2,0,1,eps)[1]))
print("Number of iterations Golden section 2nd = " + str(golden_section(f2,0,1,eps)[2]))
print("Number of f calc Golden section 2nd = " + str(golden_section(f2,0,1,eps)[3]))

graph_second = []
scale2 = []
for i in range(0,1000):
    graph_second.append(abs(i/1000-0.2))
    scale2.append(i/1000)

plt.plot(scale2[199:202],graph_second[199:202],label='|x-0.2|')
plt.plot(bruteforce(f2,0,1,eps)[0],bruteforce(f2,0,1,eps)[1],'ro',label = "bruteforce") 
plt.plot(dichotomy(f2,0,1,eps)[0],dichotomy(f2,0,1,eps)[1],'bo',label = "dichotomy") 
plt.plot(golden_section(f2,0,1,eps)[0],golden_section(f2,0,1,eps)[1],'go',label = "golden_section") 
plt.title("Algoritms comparison")
plt.legend()
plt.savefig('Plots\comp_1_2.png')
plt.show()


#Results for 1_3


print("x_min brute force 3rd = " + str(bruteforce(f3,0.01,1,eps)[0]))
print("f_min brute force 3rd = " + str(bruteforce(f3,0.01,1,eps)[1]))
print("Number of iterations brute force 3rd = " + str(bruteforce(f3,0.01,1,eps)[2]))
print("Number of f calc brute force 3rd = " + str(bruteforce(f3,0.01,1,eps)[3]))


print("x_min Dichotomy 3rd = " + str(dichotomy(f3,0.01,1,eps)[0]))
print("f_min Dichotomy 3rd = " + str(dichotomy(f3,0.01,1,eps)[1]))
print("Number of iterations Dichotomy 3rd = " + str(dichotomy(f3,0.01,1,eps)[2]))
print("Number of f calc Dichotomy 3rd = " + str(dichotomy(f3,0.01,1,eps)[3]))

print("x_min Golden section 3rd = " + str(golden_section(f3,0.01,1,eps)[0]))
print("f_min Golden section 3rd = " + str(golden_section(f3,0.01,1,eps)[1]))
print("Number of iterations Golden section 3rd = " + str(golden_section(f3,0.01,1,eps)[2]))
print("Number of f calc Golden section 3rd = " + str(golden_section(f3,0.01,1,eps)[3]))



graph_third = []
scale3 = []
for i in range(10,1000):
    i = i/1000
    graph_third.append(i*np.sin(1/i))
    scale3.append(i)

plt.plot(scale3,graph_third,label='x sin(1/x)')
plt.plot(bruteforce(f3,0.01,1,eps)[0],bruteforce(f3,0.01,1,eps)[1],'ro',label = "bruteforce") 
plt.plot(dichotomy(f3,0.01,1,eps)[0],dichotomy(f3,0.01,1,eps)[1],'bo',label = "dichotomy") 
plt.plot(golden_section(f3,0.01,1,eps)[0],golden_section(f3,0.01,1,eps)[1],'go',label = "golden_section") 
plt.title("Algoritms comparison")
plt.legend()
plt.savefig('Plots\comp_1_3_first.png')
plt.show()



plt.plot(scale3[211:218],graph_third[211:218],label='x sin(1/x)')
plt.plot(bruteforce(f3,0.01,1,eps)[0],bruteforce(f3,0.01,1,eps)[1],'ro',label = "bruteforce") 
plt.plot(dichotomy(f3,0.01,1,eps)[0],dichotomy(f3,0.01,1,eps)[1],'bo',label = "dichotomy") 
plt.plot(golden_section(f3,0.01,1,eps)[0],golden_section(f3,0.01,1,eps)[1],'go',label = "golden_section") 
plt.title("Algoritms comparison")
plt.legend()
plt.savefig('Plots\comp_1_3_second.png')
plt.show()



#2

#initialize data
alpha_real,beta_real = np.random.rand(2)

y_k = []
x_k = []
for k in range(0,101):
    x_k.append(k/100)
    y_k.append(alpha_real*k/100 + beta_real)
    #np.random.normal(loc=0,scale= 1, size=1)[0]



#BRUTEFORCE





def f12(y_k ,x_list ,var1,var2):
    #print(y_k)
    #print(x_list)
    #print(var1)
    #print(var2)
    f_pred = [var1*x+var2 for x in x_list]
    error =  [(var1 - var2)**2 for var1, var2 in zip(f_pred, y_k)]
    return sum(error)


def f22(y_k ,x_list ,var1,var2):
    f_pred = [var1/(1+x*var2) for x in x_list]
    error =  [(var1 - var2)**2 for var1, var2 in zip(f_pred, y_k)]
    return sum(error)











plt.plot(y_k)

eps = 0.001


def bruteforce_mult(fx,y_k,x_k,lb,ub,eps):

    n = int((ub-lb)/eps)
    a_min = lb
    b_min = lb
    f_calc = 0
    iter = 0
    f_x_min = fx(y_k,x_k,lb,lb)
    for k in range(int(lb*100),(n+1)*100,100):
        k = k/100
        iter += 1
        #print(iter)
        a_k = 0+k*(1-0)/n

        for k2 in range(int(lb*100),(n+1)*100,100):
            k2 = k2/100
            iter += 1
            b_k = 0+k2*(1-0)/n
            #print(str(a_k) + " - " + str(b_k))
            kth_value =f12(y_k,x_k,a_k,b_k)
            f_calc += 1
            if kth_value < f_x_min:
                f_x_min = kth_value
                a_min = a_k 
                b_min = b_k
    return a_min, b_min, f_x_min, iter, f_calc


a_min, b_min, f_x_min, iter, f_calc =  bruteforce_mult(f12,y_k,x_k,0,1,eps)





a_varianst = np.linspace(0,1,100)
b_variants = np.linspace(0,1,100)
[a_grid, b_grid] = np.meshgrid(a_varianst,b_variants)

f_grid =  f12(y_k,x_k,a_grid,b_grid)



fig = plt.figure(figsize= [12.8, 9.6])
graph = fig.add_subplot(111, projection='3d')
risunok = graph.plot_surface(a_grid,b_grid,f_grid,cmap=cm.coolwarm,alpha=0.7)
graph.scatter(a_min, b_min, f_x_min, c='red',marker='o',s=50)
graph.set_xlabel('a Label')
graph.set_ylabel('b Label')
graph.set_zlabel('Error Label')
graph.view_init(30, 135)
fig.colorbar(risunok, shrink=0.2, aspect=10)
plt.title("Grid multi first")
plt.show()


#2)

a_varianst_rational = np.linspace(0,1,100)
b_variants_rational = np.linspace(0,1,100)
[a_grid_rational, b_grid_rational] = np.meshgrid(a_varianst_rational,b_variants_rational)

f_grid_rational =  f22(y_k,x_k,a_grid_rational,b_grid_rational)


a_min2, b_min2, f_x_min2, iter2, f_calc2 =  bruteforce_mult(f22,y_k,x_k,0,1,eps)

fig = plt.figure(figsize= [12.8, 9.6])
graph = fig.add_subplot(111, projection='3d')
risunok = graph.plot_surface(a_grid_rational,b_grid_rational,f_grid_rational,cmap=cm.coolwarm,alpha=0.7)
graph.scatter(a_min2, b_min2, f_x_min2, c='red',marker='o',s=50)
graph.set_xlabel('a Label')
graph.set_ylabel('b Label')
graph.set_zlabel('Error Label')
graph.view_init(30, 135)
fig.colorbar(risunok, shrink=0.2, aspect=10)
plt.title("Grid multi first")
plt.show()


#Gauss











def dichotomy_new(fx,a,b,eps,opt_value,y_k ,x_list ,var1,var2):
    if opt_value == "var1":
        a_0 = a
        b_0 = b
        x_1 = (a_0 + b_0 - eps/20)/2
        x_2 = (a_0 + b_0 + eps/20)/2
        f_calc_dic1 = 0
        iter_dic1 = 1
        a_1 = a
        b_1 = b
        while abs(a_1 - b_1)>= eps:
            if fx(y_k ,x_list ,x_1 ,var2) <= fx(y_k ,x_list ,x_2,var2):
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
        #f_x_min_dic1 = fx(var1 = x_min_dic1)
        f_x_min_dic1 = fx(y_k ,x_list ,x_min_dic1,var2)
        return x_min_dic1, f_x_min_dic1, iter_dic1, f_calc_dic1
     
    elif opt_value == "var2":
        a_0 = a
        b_0 = b
        x_1 = (a_0 + b_0 - eps/20)/2
        x_2 = (a_0 + b_0 + eps/20)/2
        f_calc_dic1 = 0
        iter_dic1 = 1
        a_1 = a
        b_1 = b
        while abs(a_1 - b_1) >= eps:
            if fx(y_k ,x_list ,var1 ,x_1) <= fx(y_k ,x_list ,var1 ,x_2):
                a_1 = a_1
                b_1 = x_2
                
            else:
                a_1 = x_1 
                b_1 = b_1

            f_calc_dic1 += 2
            iter_dic1 +=1
            x_1 = (a_1 + b_1 - eps/2)/2
            x_2 = (a_1 + b_1 + eps/2)/2
            #print(str(a_1) + " - "+ str(b_1) )
        x_min_dic1 = (b_1 + a_1)/2
        #f_x_min_dic1 = fx(var2 = x_min_dic1)
        f_x_min_dic1 = fx(y_k ,x_list ,var1 ,x_min_dic1)

        return x_min_dic1, f_x_min_dic1, iter_dic1, f_calc_dic1

dichotomy_new(f12,0,1,0.001,'var1',y_k ,x_k ,0,0.9748271672150726)


alpha_real,beta_real

def gauss(y_k,x_k,fx,eps):
    
    alph = random.uniform(0, 1)
    beta = random.uniform(0, 1)
    f_old = 100
    f_new = 0
    iter = 0
    f_calc = 0

    #while  (abs(alph_new - alph_old)>eps)  and (abs(beta_new - beta_old)>eps) :
    while   (abs(f_new - f_old)>eps) :
        f_old = fx(y_k, x_k,var1 = alph,var2 = beta)

        #print(str(alph) + ' - ' + str(beta))

        alph1 = dichotomy_new(fx,0,1,0.000001,'var1',y_k ,x_k ,0,beta)[0]
        beta1 = dichotomy_new(fx,0,1,0.000001,'var2',y_k ,x_k ,alph1,0)[0]
        f_new = fx(y_k, x_k,alph1, beta1)
        f_calc += 1 + dichotomy_new(fx,0,1,0.000001,'var2',y_k ,x_k ,alph1,0)[3] + dichotomy_new(fx,0,1,0.000001,'var1',y_k ,x_k ,0,beta)[3]
        iter += 1 + dichotomy_new(fx,0,1,0.000001,'var2',y_k ,x_k ,alph1,0)[2] + dichotomy_new(fx,0,1,0.000001,'var1',y_k ,x_k ,0,beta)[2]
        alph = alph1
        beta = beta1
    return alph, beta, f_new,iter, f_calc
    

a_min_gauss1, b_min_gauss1, f_x_min_gauss1, iter_gauss1, f_calc_gauss1 = gauss(y_k,x_k,f12,0.001)


fig = plt.figure(figsize= [12.8, 9.6])
graph = fig.add_subplot(111, projection='3d')
risunok = graph.plot_surface(a_grid,b_grid,f_grid,cmap=cm.coolwarm,alpha=0.7)
graph.scatter(a_min_gauss1, b_min_gauss1, f_x_min_gauss1, c='red',marker='o',s=50)
graph.set_xlabel('a Label')
graph.set_ylabel('b Label')
graph.set_zlabel('Error Label')
graph.view_init(30, 135)
fig.colorbar(risunok, shrink=0.2, aspect=10)
plt.title("Gauss second")
plt.show()





gauss(y_k,x_k,f22,0.001)



a_min_gauss2, b_min_gauss2, f_x_min_gauss2, iter_gauss2, f_calc_gauss2 = gauss(y_k,x_k,f22,0.001)


fig = plt.figure(figsize= [12.8, 9.6])
graph = fig.add_subplot(111, projection='3d')
risunok = graph.plot_surface(a_grid_rational,b_grid_rational,f_grid_rational,cmap=cm.coolwarm,alpha=0.7)
graph.scatter(a_min_gauss2, b_min_gauss2, f_x_min_gauss2, c='red',marker='o',s=50)
graph.set_xlabel('a Label')
graph.set_ylabel('b Label')
graph.set_zlabel('Error Label')
graph.view_init(30, 135)
fig.colorbar(risunok, shrink=0.2, aspect=10)
plt.title("Gauss second")
plt.show()






def f12_nelder(params):
    var1,var2 = params[0], params[1]
    #print(y_k)
    #print(x_list)
    #print(var1)
    #print(var2)
    f_pred = [var1*x+var2 for x in x_k]
    error =  [(var1 - var2)**2 for var1, var2 in zip(f_pred, y_k)]
    return sum(error)


def f22_nelder(params):
    var1,var2 = params[0], params[1]
    f_pred = [var1/(1+x*var2) for x in x_k]
    error =  [(var1 - var2)**2 for var1, var2 in zip(f_pred, y_k)]
    return sum(error)



a = 0
b = 100


def nelder_mead(f, x_start,
                step=0.1, no_improve_thr=0.001,
                no_improv_break=10, 
                alpha=1., gamma=2., rho=-0.5, sigma=0.5, a=0,b=1):

    def one_val(x):
    #    if x >b:
    #        return b
    #    elif x < a :
    #        return a
    #    else:
            return x
    
    def two_val(x):
    #    for i in range(len(x)):
    #        if x[i] > b:
    #            x[i] = b
    #        elif x[i] < a:
    #            x[i] = a
        return x

    iter_count = 1
    f_calc = 0
    # init
    dim = len(x_start)
    prev_best = f(x_start)
    f_calc += 1
    no_improv = 0
    res = [[x_start, prev_best]]

    for i in range(dim):
        x = copy.copy(x_start)
        x[i] = x[i] + step
        score = f(x)
        f_calc += 1
        res.append([x, score])

    # simplex iter
    
    while 1:
        iter_count += 1
        # order
        res.sort(key=lambda x: x[1])
        best = res[0][1]

        if best < prev_best - no_improve_thr:
            no_improv = 0
            prev_best = best
        else:
            no_improv += 1

        if no_improv >= no_improv_break:
            return res[0][0][0],res[0][0][1],res[0][1] , iter_count, f_calc

        # centroid
        x0 = [0.] * dim
        for tup in res[:-1]:
            for i, c in enumerate(tup[0]):
                x0[i] += c / (len(res)-1)
                x0[i] = one_val(x0[i])

        # reflection
        xr = x0 + alpha*(x0 - res[-1][0])
        xr = two_val(xr)
        rscore = f(xr)
        f_calc += 1
        if res[0][1] <= rscore < res[-2][1]:
            del res[-1]
            res.append([xr, rscore])
            continue

        # expansion
        if rscore < res[0][1]:
            xe = x0 + gamma*(x0 - res[-1][0])
            xe = two_val(xe)
            escore = f(xe)
            f_calc += 1
            if escore < rscore:
                del res[-1]
                res.append([xe, escore])
                continue
            else:
                del res[-1]
                res.append([xr, rscore])
                continue

        # contraction
        xc = x0 + rho*(x0 - res[-1][0])
        xc = two_val(xc)
        cscore = f(xc)
        f_calc += 1
        if cscore < res[-1][1]:
            del res[-1]
            res.append([xc, cscore])
            continue

        # reduction
        x1 = res[0][0]
        nres = []
        for tup in res:
            redx = x1 + sigma*(tup[0] - x1)
            redx = two_val(redx)
            score = f(redx)
            f_calc += 1
            nres.append([redx, score])
        res = nres





nelder_mead(f22_nelder, np.array([0.0, 0.0]))







#rational

a_min2_nelder, b_min2_nelder, f_x_min2_nelder, iter2_nelder, f_calc2_nelder =  nelder_mead(f22_nelder, np.array([0.0, 0.0]))

fig = plt.figure(figsize= [12.8, 9.6])
graph = fig.add_subplot(111, projection='3d')
risunok = graph.plot_surface(a_grid_rational,b_grid_rational,f_grid_rational,cmap=cm.coolwarm,alpha=0.7)
graph.scatter(a_min2_nelder, b_min2_nelder, f_x_min2_nelder, c='red',marker='o',s=50)
graph.set_xlabel('a Label')
graph.set_ylabel('b Label')
graph.set_zlabel('Error Label')
graph.view_init(30, 55)
fig.colorbar(risunok, shrink=0.2, aspect=10)
plt.title("Grid multi first")
plt.show()