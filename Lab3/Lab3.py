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
import scipy.optimize

alpha_real,beta_real = np.random.rand(2)

y_k = []
x_k = []
for k in range(0,101):
    x_k.append(k/100)
    y_k.append(alpha_real*k/100 + beta_real)
    #np.random.normal(loc=0,scale= 1, size=1)[0]


data = pd.read_csv("data.csv")

x_k = data["x_k"]
y_k = data['y_k']
y_k_real = data['y_k_real']

plt.plot(x_k,y_k_real)
plt.scatter(x_k,y_k)





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

def rational_grad(params):
    var1,var2 = params[0], params[1]
    func_grad_var1 =  [2/(1 + var2*x) * (var1/(1+var2*x)-y) for x,y in zip(x_k, y_k)]
    func_grad_var2 = [-2*(1+var2*x)**(-2)*x *var1* (var1/(1+var2*x)-y) for x,y in zip(x_k, y_k)]
    return sum(func_grad_var1), sum(func_grad_var2)


def rational_grad2(params):
    var1,var2 = params[0], params[1]
    func_grad_var1 =  [-2*(-var1 + var2*x*y + y)/(var2*x+1)**2 for x,y in zip(x_k, y_k)]
    #func_grad_var2 = [-((2*var1*x*(var1 - y*(var2*x+1)))/(var2*x+1)**3) for x,y in zip(x_k, y_k)]
    func_grad_var2 = [-(2*var1*x*(var1-var2*x*y-y)/(var2*x+1)**3) for x,y in zip(x_k, y_k)]
    return sum(func_grad_var1), sum(func_grad_var2)


def rational_grad3(params):
    var1,var2 = params[0], params[1]
    func_grad_var1 =  [-2*(-var1 + var2*x*y + y)/(var2*x+1)**2 for x,y in zip(x_k, y_k)]
    #func_grad_var2 = [-((2*var1*x*(var1 - y*(var2*x+1)))/(var2*x+1)**3) for x,y in zip(x_k, y_k)]
    func_grad_var2 = [-(var1/(1 + var2*x) -y)*2*x*var1/(1 + var2)**2 for x,y in zip(x_k, y_k)]
    return sum(func_grad_var1), sum(func_grad_var2)



def gradient_descent(fx_grad, eta=0.0001, num_iterations=1000, a_init = 0, b_init = 0):
    
    a_list=[]
    b_list=[]
    a_list.append(a_init)
    b_list.append(b_init)
    a = a_init
    b = b_init
    f_calc = 0
    a0 = np.array([a,b])
    a_n = np.array([1,1])
    beta = 0.001

    for i in range(num_iterations):
        if i ==0:
            beta = np.array([0.001,0.001])
        else:
            beta = np.abs((a_n - a0)*(np.array(fx_grad(a_n)) - np.array(fx_grad(a0))))/np.linalg.norm((np.array(fx_grad(a_n)) - np.array(fx_grad(a0))))**2
        print(beta)
        grad_a, grad_b =  fx_grad([a,b])[0], fx_grad([a,b])[1]
        f_calc += 2
        a0 = np.array([a,b])
        a = a - eta * grad_a
        b = b - eta * grad_b
        a_n = np.array([a,b]) 
        #print(a)
        a_list.append(a)
        b_list.append(b)
       
    return a_list[-1], b_list[-1], num_iterations, f_calc










# Conjugate Gradient

def dichotomy(fx,a,b,eps):
    a_0 = a
    b_0 = b
    x_1 = (a_0 + b_0 - eps/2)/2
    x_2 = (a_0 + b_0 + eps/2)/2
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
        x_1 = (a_1 + b_1 - eps/2)/2
        x_2 = (a_1 + b_1 + eps/2)/2
        
    x_min_dic1 = (b_1 + a_1)/2
    f_x_min_dic1 = fx(x_min_dic1)
    return x_min_dic1, f_x_min_dic1, iter_dic1, f_calc_dic1



def minima_conj_linear(a):
    params = a_0 + a*s_n
    return linear(params)


def minima_conj_rational(a):
    params = a_0 + a*s_n
    return rational(params)

fx_grad = rational_grad
fx_min = minima_conj_rational


def conj_gradient(fx_grad,fx_min,a_init=1.1,b_init=-0.2,n=100):

    global a_0
    a_0 = [a_init,b_init]
    

    delta_a_0 = np.array(fx_grad(a_0)) * -1
    #for func
    global s_n
    s_n = delta_a_0
    dich = dichotomy(fx_min,-10000,10000,0.001)
    
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
        
        print(s_0)
        s_n = np.array(delta_a_n + beta*s_0)
        #print(s_n)
        dich = dichotomy(fx_min,-10000,10000,0.001)
        #print(dichotomy(fx_min,-1,1,0.001)[1]) 
        alph_0 =  dich[0]

        a_0 = np.array(a_0 + alph_0*s_n)
        #print(a_0)
        s_0 = copy.deepcopy(s_n)
        delta_a_0 = copy.deepcopy(delta_a_n)
        f_calc += dich[3]
        print(rational(a_0))
    return a_0[0],a_0[1],  iter, f_calc



#linear
conj_gradient(linear_grad, minima_conj_linear, n=100)

#rational
conj_gradient(rational_grad2, minima_conj_rational, n=8)

#packages

scipy.optimize.minimize(linear, [0,0], method = 'CG',    options={'maxiter': 1000})
scipy.optimize.minimize(rational , [0,0], method = 'CG', options={'maxiter': 1000})







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

a_init = 0
b_init = 0
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


#packages

scipy.optimize.minimize(linear, [0,0],jac =linear_grad, hess = Hessian_linear , method = 'Newton-CG',          options={'maxiter': 1000})
scipy.optimize.minimize(rational , [0,0],jac = rational_grad, hess =Hessian_rational , method = 'Newton-CG', options={'maxiter': 1000})




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

#packages


def linear_LMA(params):
    var1,var2 = params[0], params[1]
    #print(y_k)
    #print(x_list)
    #print(var1)
    #print(var2)
    f_pred = [var1*x+var2 for x in x_k]
    error =  [(var1 - var2)**2 for var1, var2 in zip(f_pred, y_k)]
    return error

def rational_LMA(params):
    var1,var2 = params[0], params[1]
    f_pred = [var1/(1+x*var2) for x in x_k]
    error =  [(var1 - var2)**2 for var1, var2 in zip(f_pred, y_k)]
    return error



scipy.optimize.least_squares(linear_LMA, [1,-0.3],loss='linear' )
scipy.optimize.least_squares(rational_LMA, [1,-0.3],loss='linear')



#RESULTS

#LINEAR


#GRADIENT DESCENT


a_grad_lin, b_grad_lin, iter_grad_lin, f_calc_grad_lin  = gradient_descent(linear_grad,eta=0.001, num_iterations=100)
f_min_grad_lin = linear([a_grad_lin, b_grad_lin])

print("Gradient descent linear a and b are: ",a_grad_lin , b_grad_lin)
print("Gradient descent linear f_min is -  ",f_min_grad_lin)
print("Gradient descent linear iter and f_calc are:  ",iter_grad_lin,f_calc_grad_lin )
grad_lin_line = [a_grad_lin*x+b_grad_lin for x in x_k]


#3d graph


a_varianst_scaled = np.linspace(0.6,0.63,100)
b_variants_scaled = np.linspace(0.89,0.91,100)
[a_grid_scaled, b_grid_scaled] = np.meshgrid(a_varianst_scaled,b_variants_scaled)
f_grid_scaled =  linear([a_grid_scaled,b_grid_scaled])

fig = plt.figure(figsize= [12.8, 9.6])
graph = fig.add_subplot(111, projection='3d')
risunok = graph.plot_surface( a_grid_scaled,b_grid_scaled,f_grid_scaled, cmap=cm.coolwarm,alpha=0.7)
graph.scatter(a_grad_lin, b_grad_lin, f_min_grad_lin, c='red',marker='o',s=50, label = "Bruteforce",alpha=1)
#graph.scatter(a_gauss_lin, b_gauss_lin, f_min_gauss_lin, c='blue',marker='o',s=50, label = "Gauss")
#graph.scatter(a_nelder_lin, b_nelder_lin, f_min_nelder_lin, c='green',marker='o',s=10, label = "Nelder-Mead")
graph.set_xlabel('a Label')
graph.set_ylabel('b Label')
graph.set_zlabel('Error Label')
graph.view_init(30, 135)
graph.legend()
fig.colorbar(risunok, shrink=0.2, aspect=10)
plt.title("Linear results comparison scaled")
#plt.savefig('Plots/3d_linear_small.png')
plt.show()



#rational

a_grad_rat, b_grad_rat, iter_grad_rat, f_calc_grad_rat  = gradient_descent(rational_grad,eta=0.001, num_iterations=30)
f_min_grad_rat = rational([a_grad_rat, b_grad_rat])

print("Gradient descent a and b are: ",a_grad_rat , b_grad_rat)
print("Gradient descent f_min is -  ",f_min_grad_rat)
print("Gradient descent iter and f_calc are:  ",iter_grad_rat,f_calc_grad_rat)
grad_rat_line = [a_grad_rat/(1+x*b_grad_rat) for x in x_k]






#3d graph
a_varianst = np.linspace(0.9,1.1,100)
b_variants = np.linspace(-0.35,-0.25,100)
[a_grid, b_grid] = np.meshgrid(a_varianst,b_variants)
f_grid =  rational([a_grid,b_grid])

fig = plt.figure(figsize= [12.8, 9.6])
graph = fig.add_subplot(111, projection='3d')
risunok = graph.plot_surface( a_grid,b_grid,f_grid, cmap=cm.coolwarm,alpha=0.7)
graph.scatter(a_grad_rat, b_grad_rat, f_min_grad_rat, c='red',marker='o',s=50, label = "Bruteforce",alpha=1)
#graph.scatter(a_gauss_rat, b_gauss_rat, f_min_gauss_rat, c='blue',marker='o',s=10, label = "Gauss")
#graph.scatter(a_nelder_rat, b_nelder_rat, f_min_nelder_rat, c='green',marker='o',s=10, label = "Nelder-Mead")
graph.set_xlabel('a Label')
graph.set_ylabel('b Label')
graph.set_zlabel('Error Label')
graph.view_init(30, 65)
graph.legend()
fig.colorbar(risunok, shrink=0.3, aspect=10)
plt.title("Rational results comparison")
#plt.savefig('Plots/3d_rat.png')
plt.show()