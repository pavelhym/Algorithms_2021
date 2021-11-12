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


alpha_real,beta_real = np.random.rand(2)

y_k = []
x_k = []
for k in range(0,101):
    x_k.append(k/100)
    y_k.append(alpha_real*k/100 + beta_real)
    #np.random.normal(loc=0,scale= 1, size=1)[0]





#Gradient Descent

def linear(params):
    var1,var2 = params[0], params[1]
    #print(y_k)
    #print(x_list)
    #print(var1)
    #print(var2)
    f_pred = [var1*x+var2 for x in x_k]
    error =  [(var1 - var2)**2 for var1, var2 in zip(f_pred, y_k)]
    return sum(error)

def linear_grad(params):
    var1,var2 = params[0], params[1]
    func_grad_var1 =  [x*2*((var1*x+var2)-y) for x,y in zip(x_k, y_k)]
    func_grad_var2 = [2*((var1*x+var2)-y) for x,y in zip(x_k, y_k)]

    return sum(func_grad_var1), sum(func_grad_var2)



def rational(params):
    var1,var2 = params[0], params[1]
    f_pred = [var1/(1+x*var2) for x in x_k]
    error =  [(var1 - var2)**2 for var1, var2 in zip(f_pred, y_k)]
    return sum(error)

def rational_grad_err(params):
    var1,var2 = params[0], params[1]
    func_grad_var1 =  [2/(1 + var2*x) * (var1/(1+var2*x)-y) for x,y in zip(x_k, y_k)]
    func_grad_var2 = [-2*(1+var2*x)**(-2)*x *(var1/(1+var2*x)-y) for x,y in zip(x_k, y_k)]
    return sum(func_grad_var1), sum(func_grad_var2)

def rational_grad(params):
    var1,var2 = params[0], params[1]
    func_grad_var1 =  [2/(1 + var2*x) * (var1/(1+var2*x)-y) for x,y in zip(x_k, y_k)]
    func_grad_var2 = [-2*(1+var2*x)**(-2)*x *var1* (var1/(1+var2*x)-y) for x,y in zip(x_k, y_k)]
    return sum(func_grad_var1), sum(func_grad_var2)



def rational_grad2(params):
    var1,var2 = params[0], params[1]
    func_grad_var1 =  [-2*(-var1 + var2*x*y + y)/(var2*x+1)**2 for x,y in zip(x_k, y_k)]
    func_grad_var2 = [-((2*var1*x*(var1 - y*(var2*x+1)))/(var2*x+1)**3) for x,y in zip(x_k, y_k)]
    return sum(func_grad_var1), sum(func_grad_var2)




#For 3d plot
a_varianst = np.linspace(0,1,100)
b_variants = np.linspace(0,1,100)
[a_grid, b_grid] = np.meshgrid(a_varianst,b_variants)
f_grid =  linear([a_grid,b_grid])

a_varianst_rational = np.linspace(0,1,100)
b_variants_rational = np.linspace(-1,0,100)
[a_grid_rational, b_grid_rational] = np.meshgrid(a_varianst_rational,b_variants_rational)
f_grid_rational =  rational([a_grid_rational,b_grid_rational])





def gradient_descent(fx_grad, eta=0.0001, num_iterations=1000, a_init = 0.5, b_init = 0.5):
    
    
    a_list=[]
    b_list=[]
    a_list.append(a_init)
    b_list.append(b_init)
    a = a_init
    b = b_init
    f_calc = 0
 
    for i in range(num_iterations):
        grad_a, grad_b =  fx_grad([a,b])[0], fx_grad([a,b])[1]
        f_calc += 2
        
        a = a - eta * grad_a
        b = b - eta * grad_b
        #print(a)
        a_list.append(a)
        b_list.append(b)
       
    return a_list[-1], b_list[-1], num_iterations, f_calc


a_min1_grad, b_min1_grad, iter1_grad, f_calc1_grad  = gradient_descent(linear_grad,eta=0.001, num_iterations=1000)
f_x_min1_gauss = linear([a_min1_grad, b_min1_grad])

fig = plt.figure()
graph = fig.add_subplot(111, projection='3d')
risunok = graph.plot_surface(a_grid,b_grid,f_grid,cmap=cm.coolwarm,alpha=0.7)
graph.scatter(a_min1_grad, b_min1_grad, f_x_min1_gauss, c='red',marker='o',s=50)
graph.set_xlabel('a Label')
graph.set_ylabel('b Label')
graph.set_zlabel('Error Label')
graph.view_init(30, 135)
fig.colorbar(risunok, shrink=0.2, aspect=10)
plt.title("Grad descent linear")
plt.show()


#rational

a_min2_grad, b_min2_grad, iter2_grad, f_calc2_grad  = gradient_descent(rational_grad,eta=0.001, num_iterations=1000)
f_x_min2_gauss = linear([a_min2_grad, b_min2_grad])



fig = plt.figure(figsize= [12.8, 9.6])
graph = fig.add_subplot(111, projection='3d')
risunok = graph.plot_surface(a_grid_rational,b_grid_rational,f_grid_rational,cmap=cm.coolwarm,alpha=0.7)
graph.scatter(a_min2_grad, b_min2_grad, f_x_min2_gauss, c='red',marker='o',s=50)
graph.set_xlabel('a Label')
graph.set_ylabel('b Label')
graph.set_zlabel('Error Label')
graph.view_init(30, 135)
fig.colorbar(risunok, shrink=0.2, aspect=10)
plt.title("Gradient decent rational")
plt.show()


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






# Conjugate Gradient

def minima_conj_linear(a):
    params = a_0 + a*s_n
    return linear(params)


def minima_conj_rational(a):
    params = a_0 + a*s_n
    return rational(params)

fx_grad = rational_grad2
fx_min = minima_conj_rational


def conj_gradient(fx_grad,fx_min,a_init=0.5,b_init=0.5,n=100):

    
    global a_0
    a_0 = [a_init,b_init]
    

    delta_a_0 = np.array(fx_grad(a_0)) * -1
    #for func
    global s_n
    s_n = delta_a_0
    dich = dichotomy(fx_min,-1000,1000,0.001)
    alph_0 =  dich[0]

    a_0 = a_0 + alph_0*delta_a_0

    s_0 = delta_a_0

    iter = n + 1
    f_calc = 2 + dich[3]

    for i in range(0,n):

        delta_a_n = np.array(fx_grad(a_0)) * -1


        #beta = delta_a_n.T.dot(delta_a_n)/(a_0.T.dot(a_0))
        #beta = delta_a_n.dot(delta_a_n.T)/(delta_a_0.dot(delta_a_0.T))

        #beta = delta_a_n.dot(delta_a_n.T - delta_a_0.T)/(delta_a_0.dot(delta_a_0.T))
        beta = 0.01
        #print(beta)
        
        #print(beta)
        
        s_n = np.array(delta_a_n + beta*s_0)
        dich = dichotomy(fx_min,-1000,1000,0.001)
        alph_0 =  dich[0]

        a_0 = np.array(a_0 + alph_0*s_n)
        print(rational(a_0))
        s_0 = s_n
        delta_a_0 = delta_a_n
        f_calc += dich[3]
    return a_0[0],a_0[1],  iter, f_calc



#linear
conj_gradient(linear_grad, minima_conj_linear, n=20)

#rational
conj_gradient(rational_grad2, minima_conj_rational,a_init=0.5,b_init=-0.5, n=15)
#DO NOT WORK




#Newton method

def Hessian_linear(params):
    var1 , var2 = params[0], params[1]
    H = np.zeros((2,2))
    H[0][0] = sum([2*x**2 for x in x_k])
    H[1][0] = sum([2*x for x in x_k])
    H[0][1] = sum([2*x for x in x_k])
    H[1][1] = 2
    return H

def Hessian_rational(params):
    var1 , var2 = params[0], params[1]
    H = np.zeros((2,2))
    H[0][0] = sum([2/(1+var2*x) for x in x_k])
    H[1][0] = sum( [x*((2*y)/(var2*x+1)**2 - (4*var1)/(var2*x+1)**3) for x,y in zip(x_k, y_k)])
    H[0][1] = sum( [x*( (2*y)/(var2*x+1)**2 - (4*var1)/(var2*x+1)**3 ) for x,y in zip(x_k, y_k)])
    H[1][1] = sum([x**2*((6*var1**2)/(var2*x+1)**4 - (4*var1*y)/(var2*x+1)**3) for x,y in zip(x_k, y_k)])
    return H

a_init = 0.5
b_init = 0.5
a_0 = [a_init,b_init]

#linear
for i in range(0,100):

   a_1 = a_0 - np.linalg.inv(Hessian_linear(a_0)).dot(np.array(linear_grad(a_0)))
   print(linear(a_1))
   a_0 = copy.deepcopy(a_1)




#Rational
#НЕ РАБОТАЕТ 
for i in range(0,10):

   a_1 = a_0 - 0.1*np.linalg.inv(Hessian_rational(a_0)).dot(np.array(rational_grad(a_0)))
   print(rational(a_1))
   a_0 = a_1



#LMA


def linear_jakob(params):
    var1,var2 = params[0], params[1]
    func_grad_var1 =  [x for x in x_k]
    func_grad_var2 = [1 for x,y in zip(x_k, y_k)]

    return np.column_stack((func_grad_var1, func_grad_var2))

def linear_LMA(params):
    var1,var2 = params[0], params[1]
    #print(y_k)
    #print(x_list)
    #print(var1)
    #print(var2)
    f_pred = [var1*x+var2 for x in x_k]
    error =  [(var1 - var2) for var1, var2 in zip(f_pred, y_k)]
    return error



beta_init = np.zeros((len(y_k),2)) + 0.5

for i in range(0,15):

    params =  [beta_init[0][0], beta_init[0][1]]
    #print(linear(params))
    jakobian = linear_jakob(params)

    error_vec = linear_LMA(params)


    LHS = jakobian.T.dot(jakobian) + 0.01*np.identity(2)

    RHS = jakobian.T.dot(error_vec)

    delta_b =  np.linalg.inv(LHS).dot(RHS)

    beta_init = beta_init + delta_b
    print(linear([beta_init[0][0],beta_init[0][1]]))

#НЕ РАБОТАЕТ 
