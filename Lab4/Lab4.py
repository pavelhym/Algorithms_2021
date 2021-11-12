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
import scipy
from scipy.optimize import minimize

#generate data 

def f(x):
    return 1/(x**2 - 3*x + 2)

y_k = []
x_k = []

for k in range(0,1001):
    x = 3*k/1000
    x_k.append(x)
    beta = 0
    if f(x) < -100:
        y_k.append(-100 + beta)
    elif (f(x) <= 100) and (f(x) >= -100):
        y_k.append(f(x) + beta)
    elif f(x) > 100:
        y_k.append(100 + beta)


plt.plot(y_k)

def D(params):
    a, b, c , d = params[0], params[1], params[2], params[3]
    f_pred = [(a*x+b)/(x**2 + c*x + d) for x in x_k]
    error =  [(pred - err)**2 for pred, err in zip(f_pred, y_k)]
    return sum(error)

def D_SE(params):
    a, b, c , d = params[0], params[1], params[2], params[3]
    f_pred = [(a*x+b)/(x**2 + c*x + d) for x in x_k]
    error =  [(pred - err)**2 for pred, err in zip(f_pred, y_k)]
    aim = [0]*len(error)
    return error

D_SE([1,1,1,1],x_k,y_k)[1]

def nelder_mead(f, x_start,
                step=0.1, no_improve_thr=0.001,
                no_improv_break=10, 
                alpha=1., gamma=2., rho=-0.5, sigma=0.5, a=0,b=1):

    def one_val(x):
        #if x >b:
        #    return b
        #elif x < a :
        #    return a
        #else:
            return x
    
    def two_val(x):
        #for i in range(len(x)):
        #    if x[i] > b:
        #        x[i] = b
        #    elif x[i] < a:
        #        x[i] = a
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
            return res[0], iter_count, f_calc

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


nelder_mead(D, np.array([0.0,1.0,-3.0,2.0]),no_improve_thr=0.001,step=0.5)


neld_mead = scipy.optimize.minimize(D, [0,1,-3,2], method = 'Nelder-Mead', options={'xatol': 0.001})

neld_mead.x
neld_mead.fun
neld_mead.nfev
neld_mead.nit



#LMA
#lin_lm = scipy.optimize.root(D_SE, [0,1,-3,2], method='lm',  options={'ftol': 0.001})

scipy.optimize.least_squares(D_SE, [1,1,-3,2], method='lm')



#Annealing


import time
import random
import math
import numpy as np
import matplotlib.pyplot as plt
#------------------------------------------------------------------------------
# Customization section:
initial_temperature = 100
cooling = 0.8  # cooling coefficient
number_variables = 4
upper_bounds = [10, 10, 10, 10]   
lower_bounds = [-10, -10, -10, -10]  
computing_time = 10 # second(s)
  

#------------------------------------------------------------------------------
# Simulated Annealing Algorithm:
initial_solution=np.zeros((number_variables))
for v in range(number_variables):
    initial_solution[v] = random.uniform(lower_bounds[v],upper_bounds[v])
      
current_solution = initial_solution
best_solution = initial_solution
n = 1  # no of solutions accepted
best_fitness = D(best_solution)
current_temperature = initial_temperature # current temperature
start = time.time()
no_attempts = 100 # number of attempts in each level of temperature
record_best_fitness =[]



for i in range(9999999):
    for j in range(no_attempts):
  
        for k in range(number_variables):
            current_solution[k] = best_solution[k] + 0.1*(random.uniform(lower_bounds[k],upper_bounds[k]))
            current_solution[k] = max((min((current_solution[k], upper_bounds[k])), lower_bounds[k]))

        current_fitness = D(current_solution)
        E = abs(current_fitness - best_fitness)
        if i == 0 and j == 0:
            EA = E
        
        if current_fitness < best_fitness:
            p = math.exp(-E/(EA * current_temperature))

            if random.random() < p:
                accept = True
            else:
                accept = False
        else:
            accept = True

        if accept == True:
            best_solution = current_solution
            best_fitness = D(best_solution)
            n = n + 1 
            EA = (EA *(n-1) + E)/n
    
    print('interation: {}, best_solution: {}, best_fitness: {}'.format(i, best_solution, best_fitness))
    record_best_fitness.append(best_fitness)

    current_temperature = current_temperature*cooling
    end = time.time()
    if end - start >= computing_time:
        break



anneal = scipy.optimize.dual_annealing(D, [(-10,10),(-10,10),(-10,10),(-10,10)])

D(anneal.x)




#Differential evolution 
from numpy.random import rand
from numpy.random import choice
from numpy import asarray
from numpy import clip
from numpy import argmin
from numpy import min
from numpy import around






bounds = asarray([(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)])


# define mutation operation
def mutation(x, F):
    return x[0] + F * (x[1] - x[2])

# define boundary check operation
def check_bounds(mutated, bounds):
    mutated_bound = [clip(mutated[i], bounds[i, 0], bounds[i, 1]) for i in range(len(bounds))]
    return mutated_bound

# define crossover operation
def crossover(mutated, target, dims, cr):
    # generate a uniform random value for every dimension
    p = rand(dims)
    # generate trial vector by binomial crossover
    trial = [mutated[i] if p[i] < cr else target[i] for i in range(dims)]
    return trial




def differential_evolution(obj, bounds,pop_size = 100, iter = 100, w = 0.5, p = 0.7):







    # initialise population of candidate solutions randomly within the specified bounds

    
    pop = bounds[:, 0] + (rand(pop_size, len(bounds)) * (bounds[:, 1] - bounds[:, 0]))
    # evaluate initial population of candidate solutions

    obj_all = [obj(ind) for ind in pop]
    # find the best performing vector of initial population
    best_vector = pop[argmin(obj_all)]
    best_obj = min(obj_all)
    prev_obj = best_obj

    f_calc = len(pop)

    # run iterations of the algorithm
    for i in range(iter):
        # iterate over all candidate solutions
        for j in range(pop_size):
            # choose three candidates, a, b and c, that are not the current one
            candidates = [candidate for candidate in range(pop_size) if candidate != j]
            a, b, c = pop[choice(candidates, 3, replace=False)]
            # perform mutation
            mutated = mutation([a, b, c], w)
            # check that lower and upper bounds are retained after mutation
            mutated = check_bounds(mutated, bounds)
            # perform crossover
            trial = crossover(mutated, pop[j], len(bounds), p)
            # compute objective function value for target vector
            obj_target = obj(pop[j])
            # compute objective function value for trial vector
            obj_trial = obj(trial)
            f_calc += 1
            # perform selection
            if obj_trial < obj_target:
                # replace the target vector with the trial vector
                pop[j] = trial
                # store the new objective function value
                obj_all[j] = obj_trial
        # find the best performing vector at each iteration
        best_obj = min(obj_all)
        # store the lowest objective function value
        if best_obj < prev_obj:
            best_vector = pop[argmin(obj_all)]
            prev_obj = best_obj
            # report progress at each iteration
            print('Iteration: %d f([%s]) = %.5f' % (i, around(best_vector, decimals=5), best_obj))
    return [best_vector, best_obj, iter,f_calc] 



result = differential_evolution(D,bounds)


pop_size = 100, iter = 100, w = 0.5, p = 0.7


dif_evol_result = scipy.optimize.differential_evolution(D, bounds, strategy='best1bin', maxiter=100, popsize=100, tol=0.01, mutation=(0.5, 1), recombination=0.7)
D(result[0])



#2



""" Traveling salesman problem solved using Simulated Annealing.
"""
from scipy import *
from pylab import *



def Distance(R1, R2):
    return np.sqrt((R1[0]-R2[0])**2+(R1[1]-R2[1])**2)


from math import sin, cos, sqrt, atan2, radians


def Distance(R1, R2):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = math.radians(R1[0])
    lon1 = math.radians(R1[1])
    lat2 = math.radians(R2[0])
    lon2 = math.radians(R2[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


def TotalDistance(city, R):
    dist=0
    for i in range(len(city)-1):
        dist += Distance(R[city[i]],R[city[i+1]])
    dist += Distance(R[city[-1]],R[city[0]])
    return dist
    
def reverse(city, n):
    nct = len(city)
    nn = (1+ ((n[1]-n[0]) % nct))/2 # half the lenght of the segment to be reversed
    # the segment is reversed in the following way n[0]<->n[1], n[0]+1<->n[1]-1, n[0]+2<->n[1]-2,...
    # Start at the ends of the segment and swap pairs of cities, moving towards the center.
    print(nn)
    for j in range(0,int(nn)):
        print(j)
        k = (n[0]+j) % nct
        l = (n[1]-j) % nct
        (city[k],city[l]) = (city[l],city[k])  # swap


    
def transpt(city, n):
    nct = len(city)
    
    newcity=[]
    # Segment in the range n[0]...n[1]
    for j in range( (n[1]-n[0])%nct + 1):
        newcity.append(city[ (j+n[0])%nct ])
    # is followed by segment n[5]...n[2]
    for j in range( (n[2]-n[5])%nct + 1):
        newcity.append(city[ (j+n[5])%nct ])
    # is followed by segment n[3]...n[4]
    for j in range( (n[4]-n[3])%nct + 1):
        newcity.append(city[ (j+n[3])%nct ])
    return newcity

def Plot(city, R, dist):
    # Plot
    Pt = [R[city[i]] for i in range(len(city))]
    Pt += [R[city[0]]]
    Pt = array(Pt)
    title('Total distance='+str(dist))
    plot(Pt[:,0], Pt[:,1], '-o')
    show()

def Plot(city, R, dist):
    # Plot
    Pt = [R[city[i]] for i in range(len(city))]
    Pt += [R[city[0]]]
    Pt = array(Pt)
    fig = px.line_mapbox( lat=Pt[:,0], lon=Pt[:,1], zoom=10, height=300)

    fig.update_layout(mapbox_style="stamen-terrain", mapbox_zoom=2.5, mapbox_center_lat = 39,mapbox_center_lon = -97,
        margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()

    print('Total distance='+str(dist))



def TSP(data):

    if __name__=='__main__':

        ncity = len(data)      # Number of cities to visit
        maxTsteps = 100    # Temperature is lowered not more than maxTsteps
        Tstart = 0.2       # Starting temperature - has to be high enough
        fCool = 0.9        # Factor to multiply temperature at each cooling step
        maxSteps = 100*ncity     # Number of steps at constant temperature
        maxAccepted = 10*ncity   # Number of accepted steps at constant temperature

        Preverse = 0.5      # How often to choose reverse/transpose trial move

        # Choosing city coordinates
        #R=[]  # coordinates of cities are choosen randomly
        #for i in range(ncity):
        #    R.append( [rand(),rand()] )
        #R = array(R)
        R = data.to_numpy()

        # The index table -- the order the cities are visited.
        city = list(range(ncity))
        # Distance of the travel at the beginning
        dist = TotalDistance(city, R)

        # Stores points of a move
        n = zeros(6, dtype=int)
        nct = len(R) # number of cities

        T = Tstart # temperature

        Plot(city, R, dist)

        for t in range(maxTsteps):  # Over temperature

            accepted = 0
            for i in range(maxSteps): # At each temperature, many Monte Carlo steps

                while True: # Will find two random cities sufficiently close by
                    # Two cities n[0] and n[1] are choosen at random
                    n[0] = int((nct)*rand())     # select one city
                    n[1] = int((nct-1)*rand())   # select another city, but not the same
                    if (n[1] >= n[0]): n[1] += 1   #
                    if (n[1] < n[0]): (n[0],n[1]) = (n[1],n[0]) # swap, because it must be: n[0]<n[1]
                    nn = (n[0]+nct -n[1]-1) % nct  # number of cities not on the segment n[0]..n[1]
                    if nn>=3: break

                # We want to have one index before and one after the two cities
                # The order hence is [n2,n0,n1,n3]
                n[2] = (n[0]-1) % nct  # index before n0  -- see figure in the lecture notes
                n[3] = (n[1]+1) % nct  # index after n2   -- see figure in the lecture notes

                if Preverse > rand(): 
                    # Here we reverse a segment
                    # What would be the cost to reverse the path between city[n[0]]-city[n[1]]?
                    de = Distance(R[city[n[2]]],R[city[n[1]]]) + Distance(R[city[n[3]]],R[city[n[0]]]) - Distance(R[city[n[2]]],R[city[n[0]]]) - Distance(R[city[n[3]]],R[city[n[1]]])

                    if de<0 or np.exp(-de/T)>rand(): # Metropolis
                        accepted += 1
                        dist += de
                        reverse(city, n)
                else:
                    # Here we transpose a segment
                    nc = (n[1]+1+ int(rand()*(nn-1)))%nct  # Another point outside n[0],n[1] segment. See picture in lecture nodes!
                    n[4] = nc
                    n[5] = (nc+1) % nct

                    # Cost to transpose a segment
                    de = -Distance(R[city[n[1]]],R[city[n[3]]]) - Distance(R[city[n[0]]],R[city[n[2]]]) - Distance(R[city[n[4]]],R[city[n[5]]])
                    de += Distance(R[city[n[0]]],R[city[n[4]]]) + Distance(R[city[n[1]]],R[city[n[5]]]) + Distance(R[city[n[2]]],R[city[n[3]]])

                    if de<0 or np.exp(-de/T)>rand(): # Metropolis
                        accepted += 1
                        dist += de
                        city = transpt(city, n)

                if accepted > maxAccepted: break

            # Plot
            Plot(city, R, dist)

            print("T=%10.5f , distance= %10.5f , accepted steps= %d" %(T, dist, accepted))
            T *= fCool             # The system is cooled down
            if accepted == 0: break  # If the path does not want to change any more, we can stop


        Plot(city, R, dist)
    return(city, R, dist)









res =  TSP(top_20)

R = res[1]
city = res[0]

import plotly.express as px
import pandas as pd





import pandas as pd

us_cities = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/us-cities-top-1k.csv")
us_cities = us_cities.sort_values(by=['Population'],ascending=False).groupby("State").first().sort_values(by=['Population'],ascending=False).head(30)

top_20 = us_cities.reset_index()[['lat','lon']]

import plotly.express as px

import plotly.express as px


plot(Pt[:,0], Pt[:,1], '-o')

fig = px.line_mapbox( lat=Pt[:,0], lon=Pt[:,1], zoom=10, height=300)

fig.update_layout(mapbox_style="stamen-terrain", mapbox_zoom=2.5, mapbox_center_lat = 39,mapbox_center_lon = -97,
    margin={"r":0,"t":0,"l":0,"b":0})
fig.show()