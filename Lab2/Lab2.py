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
alpha_real,beta_real = 0.6236249972188823 , 0.8160876383416222

y_k_real = []
y_k = []
x_k = []
for k in range(0,101):
    x_k.append(k/100)
    noise = 0
    noise = np.random.normal(loc=0,scale= 1, size=1)[0]
    y_k.append(alpha_real*k/100 + beta_real + noise)
    y_k_real.append(alpha_real*k/100 + beta_real)

random_scatter = pd.DataFrame()
random_scatter["x_k"] = x_k
random_scatter["y_k"] = y_k
random_scatter["y_k_real"] = y_k_real
#random_scatter.to_csv("data.csv")

data = pd.read_csv("data.csv")

x_k = data["x_k"]
y_k = data['y_k']
y_k_real = data['y_k_real']

plt.plot(x_k,y_k_real)
plt.scatter(x_k,y_k)


#BRUTEFORCE





def linear(y_k ,x_list ,var1,var2):
    #print(y_k)
    #print(x_list)
    #print(var1)
    #print(var2)
    f_pred = [var1*x+var2 for x in x_list]
    error =  [(var1 - var2)**2 for var1, var2 in zip(f_pred, y_k)]
    return sum(error)


def rational(y_k ,x_list ,var1,var2):
    f_pred = [var1/(1+x*var2) for x in x_list]
    error =  [(var1 - var2)**2 for var1, var2 in zip(f_pred, y_k)]
    return sum(error)









eps = 0.001


def bruteforce_mult(fx,y_k,x_k,lb,ub,eps):

    n = int((ub-lb)/eps)
    a_min = lb
    b_min = lb
    f_calc = 0
    iter = 0
    f_x_min = fx(y_k,x_k,lb,ub)
    for k in range(int(lb*100),(n+1)*100,100):
        k = k/100
        
        #print(iter)
        a_k = lb+k*(ub-lb)/n

        for k2 in range(int(lb*100),(n+1)*100,100):
            k2 = k2/100
            
            b_k = lb+k2*(ub-lb)/n
            b_k == -1
            
            #print(str(a_k) + " - " + str(b_k))
            try:
                kth_value =fx(y_k,x_k,a_k,b_k)
            except ZeroDivisionError:
                continue
            #kth_value =fx(y_k,x_k,a_k,b_k)
            f_calc += 1
            iter += 1
            if kth_value < f_x_min:
                f_x_min = kth_value
                a_min = a_k 
                b_min = b_k
    return a_min, b_min, f_x_min, iter, f_calc



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



def gauss(y_k,x_k,fx,lb,ub,eps):
    
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

        alph1 = dichotomy_new(fx,lb,ub,eps,'var1',y_k ,x_k ,0,beta)[0]
        beta1 = dichotomy_new(fx,lb,ub,eps,'var2',y_k ,x_k ,alph1,0)[0]
        f_new = fx(y_k, x_k,alph1, beta1)
        f_calc += 1 + dichotomy_new(fx,lb,ub,eps,'var2',y_k ,x_k ,alph1,0)[3] + dichotomy_new(fx,lb,ub,eps,'var1',y_k ,x_k ,0,beta)[3]
        iter += 1 + dichotomy_new(fx,lb,ub,eps,'var2',y_k ,x_k ,alph1,0)[2] + dichotomy_new(fx,lb,ub,eps,'var1',y_k ,x_k ,0,beta)[2]
        alph = alph1
        beta = beta1
    return alph, beta, f_new,iter, f_calc
    

def linear_nelder(params):
    var1,var2 = params[0], params[1]
    #print(y_k)
    #print(x_list)
    #print(var1)
    #print(var2)
    f_pred = [var1*x+var2 for x in x_k]
    error =  [(var1 - var2)**2 for var1, var2 in zip(f_pred, y_k)]
    return sum(error)


def rational_nelder(params):
    var1,var2 = params[0], params[1]
    f_pred = [var1/(1+x*var2) for x in x_k]
    error =  [(var1 - var2)**2 for var1, var2 in zip(f_pred, y_k)]
    return sum(error)




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

    iter_count = 0
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




#Linear Results

a_brut_lin, b_brut_lin, f_min_brut_lin, iter_brut_lin, f_calc_brut_lin =  bruteforce_mult(linear,y_k,x_k,0,1,eps)
print("Bruteforce linear a and b are: ",a_brut_lin , b_brut_lin)
print("Bruteforce linear f_min is -  ",f_min_brut_lin)
print("Bruteforce linear iter and f_calc are:  ",iter_brut_lin,f_calc_brut_lin )
brut_lin_line = [a_brut_lin*x+b_brut_lin for x in x_k]



a_gauss_lin, b_gauss_lin, f_min_gauss_lin, iter_gauss_lin, f_calc_gauss_lin = gauss(y_k,x_k,linear,0,1,0.001)
print("Gauss linear a and b are: ",a_gauss_lin , b_gauss_lin)
print("Gauss linear f_min is -  ",f_min_gauss_lin)
print("Gauss linear iter and f_calc are:  ",iter_gauss_lin,f_calc_gauss_lin )
gauss_lin_line = [a_gauss_lin*x+b_gauss_lin for x in x_k]



a_nelder_lin, b_nelder_lin, f_min_nelder_lin, iter_nelder_lin, f_calc_nelder_lin = nelder_mead(linear_nelder, np.array([0.0, 0.0]))
print("Nelder linear a and b are: ",a_nelder_lin , b_nelder_lin)
print("Nelder linear f_min is -  ",f_min_nelder_lin)
print("Nelder linear iter and f_calc are:  ",iter_nelder_lin,f_calc_nelder_lin )
nelder_lin_line = [a_nelder_lin*x+b_nelder_lin for x in x_k]


#3d graphs

a_varianst = np.linspace(0,1,100)
b_variants = np.linspace(0,1,100)
[a_grid, b_grid] = np.meshgrid(a_varianst,b_variants)
f_grid =  linear(y_k,x_k,a_grid,b_grid)

fig = plt.figure(figsize= [12.8, 9.6])
graph = fig.add_subplot(111, projection='3d')
risunok = graph.plot_surface( a_grid,b_grid,f_grid, cmap=cm.coolwarm,alpha=0.7)
graph.scatter(a_brut_lin, b_brut_lin, f_min_brut_lin, c='red',marker='o',s=50, label = "Bruteforce",alpha=1)
graph.scatter(a_gauss_lin, b_gauss_lin, f_min_gauss_lin, c='blue',marker='o',s=10, label = "Gauss")
graph.scatter(a_nelder_lin, b_nelder_lin, f_min_nelder_lin, c='green',marker='o',s=10, label = "Nelder-Mead")
graph.set_xlabel('a Label')
graph.set_ylabel('b Label')
graph.set_zlabel('Error Label')
graph.view_init(30, 135)
graph.legend()
fig.colorbar(risunok, shrink=0.2, aspect=10)
plt.title("Linear results comparison")
plt.savefig('Plots/3d_linear_big.png')
plt.show()




a_varianst_scaled = np.linspace(0.6,0.63,100)
b_variants_scaled = np.linspace(0.89,0.91,100)
[a_grid_scaled, b_grid_scaled] = np.meshgrid(a_varianst_scaled,b_variants_scaled)
f_grid_scaled =  linear(y_k,x_k,a_grid_scaled,b_grid_scaled)

fig = plt.figure(figsize= [12.8, 9.6])
graph = fig.add_subplot(111, projection='3d')
risunok = graph.plot_surface( a_grid_scaled,b_grid_scaled,f_grid_scaled, cmap=cm.coolwarm,alpha=0.7)
graph.scatter(a_brut_lin, b_brut_lin, f_min_brut_lin, c='red',marker='o',s=50, label = "Bruteforce",alpha=1)
graph.scatter(a_gauss_lin, b_gauss_lin, f_min_gauss_lin, c='blue',marker='o',s=50, label = "Gauss")
graph.scatter(a_nelder_lin, b_nelder_lin, f_min_nelder_lin, c='green',marker='o',s=10, label = "Nelder-Mead")
graph.set_xlabel('a Label')
graph.set_ylabel('b Label')
graph.set_zlabel('Error Label')
graph.view_init(30, 135)
graph.legend()
fig.colorbar(risunok, shrink=0.2, aspect=10)
plt.title("Linear results comparison scaled")
plt.savefig('Plots/3d_linear_small.png')
plt.show()


#scatter graphs


plt.plot(x_k, y_k_real, label = "no noise data", linewidth=7.0, alpha = 0.7)
plt.scatter(x_k,y_k,label = 'noisy data')
plt.plot(x_k, brut_lin_line, label = 'Brut-force line'   ,linewidth=4.0)
plt.plot(x_k, gauss_lin_line, label = 'Gauss line',linewidth=3.0)
plt.plot(x_k, nelder_lin_line, label = 'Nelder-Mead line',linewidth=1.0)
plt.legend()
plt.title("Linear approximations")
plt.savefig('Plots/scat_linear.png')
plt.show()

#Rational results


a_brut_rat, b_brut_rat, f_min_brut_rat, iter_brut_rat, f_calc_brut_rat =  bruteforce_mult(rational,y_k,x_k,-0.4,1,eps)
print("Bruteforce rational a and b are: ",a_brut_rat , b_brut_rat)
print("Bruteforce rational f_min is -  ",f_min_brut_rat)
print("Bruteforce rational iter and f_calc are:  ",iter_brut_rat,f_calc_brut_rat )
brut_rat_line = [a_brut_rat/(1+x*b_brut_rat) for x in x_k]


a_gauss_rat, b_gauss_rat, f_min_gauss_rat, iter_gauss_rat, f_calc_gauss_rat = gauss(y_k,x_k,rational,-2,2,0.001)
print("Gauss rational a and b are: ",a_gauss_rat , b_gauss_rat)
print("Gauss rational f_min is -  ",f_min_gauss_rat)
print("Gauss rational iter and f_calc are:  ",iter_gauss_rat,f_calc_gauss_rat)
gauss_rat_line = [a_gauss_rat/(1+x*b_gauss_rat) for x in x_k]


a_nelder_rat, b_nelder_rat, f_min_nelder_rat, iter_nelder_rat, f_calc_nelder_rat = nelder_mead(rational_nelder, np.array([0.0, 0.0]))
print("Nelder rational a and b are: ",a_nelder_rat , b_nelder_rat)
print("Nelder rational f_min is -  ",f_min_nelder_rat)
print("Nelder rational iter and f_calc are:  ",iter_nelder_rat,f_calc_nelder_rat)
nelder_rat_line = [a_nelder_rat/(1+x*b_nelder_rat) for x in x_k]




#3d graph
a_varianst = np.linspace(0.9,1.1,100)
b_variants = np.linspace(-0.35,-0.25,100)
[a_grid, b_grid] = np.meshgrid(a_varianst,b_variants)
f_grid =  rational(y_k,x_k,a_grid,b_grid)

fig = plt.figure(figsize= [12.8, 9.6])
graph = fig.add_subplot(111, projection='3d')
risunok = graph.plot_surface( a_grid,b_grid,f_grid, cmap=cm.coolwarm,alpha=0.7)
graph.scatter(a_brut_rat, b_brut_rat, f_min_brut_rat, c='red',marker='o',s=50, label = "Bruteforce",alpha=1)
graph.scatter(a_gauss_rat, b_gauss_rat, f_min_gauss_rat, c='blue',marker='o',s=10, label = "Gauss")
graph.scatter(a_nelder_rat, b_nelder_rat, f_min_nelder_rat, c='green',marker='o',s=10, label = "Nelder-Mead")
graph.set_xlabel('a Label')
graph.set_ylabel('b Label')
graph.set_zlabel('Error Label')
graph.view_init(30, 65)
graph.legend()
fig.colorbar(risunok, shrink=0.3, aspect=10)
plt.title("Rational results comparison")
plt.savefig('Plots/3d_rat.png')
plt.show()


# scatter graphs

plt.plot(x_k, y_k_real, label = "no noise data", linewidth=7.0, alpha = 0.7)
plt.scatter(x_k,y_k,label = 'noisy data')
plt.plot(x_k, brut_rat_line, label = 'Brut-force line'   ,linewidth=4.0)
plt.plot(x_k, gauss_rat_line, label = 'Gauss line'       ,linewidth=3.0)
plt.plot(x_k, nelder_rat_line, label = 'Nelder-Mead line',linewidth=1.0)
plt.legend()
plt.title("Rational approximations")
plt.savefig('Plots/scat_rat.png')
plt.show()


