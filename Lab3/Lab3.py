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

"""Firstly I want to honestly claim, that I've tried very hard to design all methods
manually, but it seems to me ureal or I am just dumb. As a result I have:
1) manual gradient descent - works properly
2) manual conjugate gradient - works properly only for linear
for rational it makes some strange things (I've checked the gradient function and it seems
to be correct)
3) Newton method - do not work and I really have no idea why, as all functions are 
made as it was said in lecture
4) LMA - do not work at all, the same problem as in Newton method
all code presented below and if you will contact me to say where are mistakes - I will be very happy
"""



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



def rational_grad(params):
    var1,var2 = params[0], params[1]
    func_grad_var1 =  [2/(1 + var2*x) * (var1/(1+var2*x)-y) for x,y in zip(x_k, y_k)]
    func_grad_var2 = [-2*(1+var2*x)**(-2)*x *var1* (var1/(1+var2*x)-y) for x,y in zip(x_k, y_k)]
    return sum(func_grad_var1), sum(func_grad_var2)




def gradient_descent(fx_grad, eta=0.0001, num_iterations=1000, a_init = 1, b_init = 1, eps = 0.001):
    
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
        grad_a, grad_b =  fx_grad([a,b])[0], fx_grad([a,b])[1]
        f_calc += 2
        a0 = np.array([a,b])
        a = a - beta[0] * grad_a
        b = b - beta[1] * grad_b
        a_n = np.array([a,b]) 
        if all(np.abs(a_n - a0) < eps):
            num_iter = i +1
            break
        #print(a)
        a_list.append(a)
        b_list.append(b)
        num_iter = i +1
       
    return a_list[-1], b_list[-1], num_iter, f_calc



#CNOJUGATE GRADIENT

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
        beta = delta_a_n.dot(delta_a_n.T)/(delta_a_0.dot(delta_a_0.T))
        #beta = delta_a_n.dot(delta_a_n.T - delta_a_0.T)/(delta_a_0.dot(delta_a_0.T))
        #beta = 0.01
        #print(beta)
        s_n = np.array(delta_a_n + beta*s_0)
        dich = dichotomy(fx_min,-10000,10000,0.001)
        alph_0 =  dich[0]
        a_0 = np.array(a_0 + alph_0*s_n)
        s_0 = copy.deepcopy(s_n)
        delta_a_0 = copy.deepcopy(delta_a_n)
        f_calc += dich[3]
        
    return a_0[0],a_0[1],  iter, f_calc

#linear
conj_gradient(linear_grad, minima_conj_linear, n=30)

#rational
# DO NOT WORK, IDK WHY
conj_gradient(rational_grad, minima_conj_rational, n=10)








#NEWTON
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
#Do not work
for i in range(0,100):

   a_1 = a_0 - np.linalg.inv(Hessian_linear(a_0)).dot(np.array(linear_grad(a_0)))
   print(linear(a_1))
   a_0 = copy.deepcopy(a_1)


#Rational
#Do not work
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

#DO NOT WORK

#for packages


def linear_LMA(params):
    var1,var2 = params[0], params[1]
    #print(y_k)
    #print(x_list)
    #print(var1)
    #print(var2)
    f_pred = [var1*x+var2 for x in x_k]
    error =  [(var1 - var2)**2 for var1, var2 in zip(f_pred, y_k)]
    return [sum(error), 0]

def rational_LMA(params):
    var1,var2 = params[0], params[1]
    f_pred = [var1/(1+x*var2) for x in x_k]
    error =  [(var1 - var2)**2 for var1, var2 in zip(f_pred, y_k)]
    return [sum(error), 0]








#RESULTS




#LINEAR


#GRADIENT DESCENT


a_grad_lin, b_grad_lin, iter_grad_lin, f_calc_grad_lin  = gradient_descent(linear_grad,eta=0.001, num_iterations=300,eps = 0.000001)
f_min_grad_lin = linear([a_grad_lin, b_grad_lin])

print("Gradient descent linear a and b are: ",a_grad_lin , b_grad_lin)
print("Gradient descent linear f_min is -  ",f_min_grad_lin)
print("Gradient descent linear iter and f_calc are:  ",iter_grad_lin,f_calc_grad_lin )
grad_lin_line = [a_grad_lin*x+b_grad_lin for x in x_k]


#conj_grad
conj_grad_results_lin = scipy.optimize.minimize(linear, [0,0], method = 'CG',    options={'maxiter': 1000})
a_cgrad_lin, b_cgrad_lin = conj_grad_results_lin.x
iter_cgrad_lin, f_calc_cgrad_lin = conj_grad_results_lin.nit, conj_grad_results_lin.nfev
f_min_cgrad_lin = linear([a_cgrad_lin, b_cgrad_lin])

print("Conjugate gradient descent linear a and b are: ",a_cgrad_lin , b_cgrad_lin)
print("Conjugate gradient descent linear f_min is -  ",f_min_cgrad_lin)
print("Conjugate gradient descent linear iter and f_calc are:  ",iter_cgrad_lin,f_calc_cgrad_lin )
cgrad_lin_line = [a_cgrad_lin*x+b_cgrad_lin for x in x_k]

#Newton

newton_results_lin =  scipy.optimize.minimize(linear, [0,0],jac =linear_grad, hess = Hessian_linear , method = 'Newton-CG',          options={'maxiter': 1000})
a_newt_lin, b_newt_lin = newton_results_lin.x
iter_newt_lin, f_calc_newt_lin = newton_results_lin.nit, newton_results_lin.nfev
f_min_newt_lin = linear([a_newt_lin, b_newt_lin])

print("Newton linear a and b are: ",a_newt_lin , b_newt_lin)
print("Newton linear f_min is -  ",f_min_newt_lin)
print("Newton linear iter and f_calc are:  ",iter_newt_lin,f_calc_newt_lin )
newt_lin_line = [a_newt_lin*x+b_newt_lin for x in x_k]



#LMA
LMA_results_lin = scipy.optimize.least_squares(linear_LMA, [1,-0.3],loss='linear',method='lm')
a_LMA_lin, b_LMA_lin = LMA_results_lin.x
iter_LMA_lin, f_calc_LMA_lin = 0, LMA_results_lin.nfev
f_min_LMA_lin = linear([a_LMA_lin, b_LMA_lin])

print("LMA linear a and b are: ",a_LMA_lin , b_LMA_lin)
print("LMA linear f_min is -  ",f_min_LMA_lin)
print("LMA linear iter and f_calc are:  ",iter_LMA_lin,f_calc_LMA_lin )
LMA_lin_line = [a_LMA_lin*x+b_LMA_lin for x in x_k]







#3d graph


a_varianst_scaled = np.linspace(0.6,0.63,100)
b_variants_scaled = np.linspace(0.89,0.91,100)
[a_grid_scaled, b_grid_scaled] = np.meshgrid(a_varianst_scaled,b_variants_scaled)
f_grid_scaled =  linear([a_grid_scaled,b_grid_scaled])

fig = plt.figure(figsize= [12.8, 9.6])
graph = fig.add_subplot(111, projection='3d')
risunok = graph.plot_surface( a_grid_scaled,b_grid_scaled,f_grid_scaled, cmap=cm.coolwarm,alpha=0.7)
graph.scatter(a_grad_lin, b_grad_lin, f_min_grad_lin, c='red',marker='o',s=50, label = "Gradient",alpha=1)
graph.scatter(a_cgrad_lin, b_cgrad_lin, f_min_cgrad_lin, c='blue',marker='o',s=10, label = "C_Gradient")
graph.scatter(a_newt_lin, b_newt_lin, f_min_newt_lin, c='green',marker='o',s=10, label = "Newton")
graph.scatter(a_LMA_lin, b_LMA_lin, f_min_LMA_lin, c='black',marker='o',s=30, label = "LMA")
graph.set_xlabel('a Label')
graph.set_ylabel('b Label')
graph.set_zlabel('Error Label')
graph.view_init(30, 135)
graph.legend()
fig.colorbar(risunok, shrink=0.2, aspect=10)
plt.title("Linear results comparison scaled")
#plt.savefig('Plots/3d_linear_small.png')
plt.show()


#Scatter
plt.plot(x_k, y_k_real, label = "no noise data", linewidth=7.0, alpha = 0.7)
plt.scatter(x_k,y_k,label = 'noisy data')
plt.plot(x_k, grad_lin_line, label = 'Gradient'   ,linewidth=4.0)
plt.plot(x_k, cgrad_lin_line, label = 'C_Gradient',linewidth=3.0)
plt.plot(x_k, newt_lin_line, label = 'Newton',linewidth=1.0)
plt.plot(x_k, LMA_lin_line, label = 'LMA',linewidth=1.0)

plt.legend()
plt.title("Linear approximations")
#plt.savefig('Plots/scat_linear.png')
plt.show()



#rational

a_grad_rat, b_grad_rat, iter_grad_rat, f_calc_grad_rat  = gradient_descent(rational_grad,eta=0.001, num_iterations=300)
f_min_grad_rat = rational([a_grad_rat, b_grad_rat])

print("Gradient descent rational a and b are: ",a_grad_rat , b_grad_rat)
print("Gradient descent rational f_min is -  ",f_min_grad_rat)
print("Gradient descent rational iter and f_calc are:  ",iter_grad_rat,f_calc_grad_rat)
grad_rat_line = [a_grad_rat/(1+x*b_grad_rat) for x in x_k]



conj_grad_results_rat = scipy.optimize.minimize(rational , [0,0], method = 'CG', options={'maxiter': 1000})
a_cgrad_rat, b_cgrad_rat = conj_grad_results_rat.x
iter_cgrad_rat, f_calc_cgrad_rat = conj_grad_results_rat.nit, conj_grad_results_rat.nfev
f_min_cgrad_rat = rational([a_cgrad_rat, b_cgrad_rat])

print("Conjugate gradient descent rational a and b are: ",a_cgrad_rat , b_cgrad_rat)
print("Conjugate gradient descent rational f_min is -  ",f_min_cgrad_rat)
print("Conjugate gradient descent rational iter and f_calc are:  ",iter_cgrad_rat,f_calc_cgrad_rat )
cgrad_rat_line = [a_cgrad_rat/(1+x*b_cgrad_rat) for x in x_k]




newton_results_rat = scipy.optimize.minimize(rational , [0,0],jac = rational_grad, hess =Hessian_rational , method = 'Newton-CG', options={'maxiter': 1000})
a_newt_rat, b_newt_rat = newton_results_rat.x
iter_newt_rat, f_calc_newt_rat = newton_results_rat.nit, newton_results_rat.nfev
f_min_newt_rat = rational([a_newt_rat, b_newt_rat])

print("Newton rational a and b are: ",a_newt_rat , b_newt_rat)
print("Newton rational f_min is -  ",f_min_newt_rat)
print("Newton rational iter and f_calc are:  ",iter_newt_rat,f_calc_newt_rat )
newt_rat_line = [a_newt_rat/(1+x*b_newt_rat) for x in x_k]



LMA_results_rat =  scipy.optimize.least_squares(rational_LMA, [0,0],loss='linear',method='lm', ftol=1e-10, xtol=1e-10, gtol=1e-10)
a_LMA_rat, b_LMA_rat = LMA_results_rat.x
iter_LMA_rat, f_calc_LMA_rat = 0, LMA_results_rat.nfev
f_min_LMA_rat = rational([a_LMA_rat, b_LMA_rat])

print("LMA rational a and b are: ",a_LMA_rat , b_LMA_rat)
print("LMA rational f_min is -  ",f_min_LMA_rat)
print("LMA rational iter and f_calc are:  ",iter_LMA_rat,f_calc_LMA_rat )
LMA_rat_line = [a_LMA_rat/(1+x*b_LMA_rat) for x in x_k]


#3d graph
a_varianst = np.linspace(0.9,1.1,100)
b_variants = np.linspace(-0.35,-0.25,100)
[a_grid, b_grid] = np.meshgrid(a_varianst,b_variants)
f_grid =  rational([a_grid,b_grid])

fig = plt.figure(figsize= [12.8, 9.6])
graph = fig.add_subplot(111, projection='3d')
risunok = graph.plot_surface( a_grid,b_grid,f_grid, cmap=cm.coolwarm,alpha=0.7)
graph.scatter(a_grad_rat, b_grad_rat, f_min_grad_rat, c='red',marker='o',s=50, label = "Gradient",alpha=1)
graph.scatter(a_cgrad_rat, b_cgrad_rat, f_min_cgrad_rat, c='blue',marker='o',s=10, label = "C_Gradient")
graph.scatter(a_newt_rat, b_newt_rat, f_min_newt_rat, c='green',marker='o',s=10, label = "Newton")
graph.scatter(a_LMA_rat, b_LMA_rat, f_min_LMA_rat, c='black',marker='o',s=10, label = "LMA")
graph.set_xlabel('a Label')
graph.set_ylabel('b Label')
graph.set_zlabel('Error Label')
graph.view_init(30, 65)
graph.legend()
fig.colorbar(risunok, shrink=0.3, aspect=10)
plt.title("Rational results comparison")
#plt.savefig('Plots/3d_rat.png')
plt.show()



plt.plot(x_k, y_k_real, label = "no noise data", linewidth=7.0, alpha = 0.7)
plt.scatter(x_k,y_k,label = 'noisy data')
plt.plot(x_k, grad_rat_line, label = 'Gradient'   ,linewidth=4.0)
plt.plot(x_k, cgrad_rat_line, label = 'C_Gradient'       ,linewidth=3.0)
plt.plot(x_k, newt_rat_line, label = 'Newton',linewidth=1.0)
plt.plot(x_k, LMA_rat_line, label = 'LMA',linewidth=1.0)

plt.legend()
plt.title("Rational approximations")
#plt.savefig('Plots/scat_rat.png')
plt.show()