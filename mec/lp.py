import pandas as pd
import numpy as np
import scipy.sparse as spr
#import gurobipy as grb
import sympy
from sympy.solvers import solve
from sympy import *
import matplotlib.pyplot as plt
from tabulate import tabulate


#############################
# LP1: Intro to linear programming #
#############################

def load_stigler_data(verbose=False):
    import pandas as pd
    thepath = 'https://raw.githubusercontent.com/math-econ-code/mec_optim_2021-01/master/data_mec_optim/lp_stigler-diet/'
    filename = 'StiglerData1939.txt'
    thedata = pd.read_csv(thepath + filename, sep='\t')
    thedata = thedata.dropna(how = 'all')
    commodities = (thedata['Commodity'].values)[:-1]
    allowance = thedata.iloc[-1, 4:].fillna(0).transpose()
    if verbose:
        print('Daily nutrient content:')
        print(tabulate(thedata.head()))
        print('\nDaily nutrient requirement:')
        print(allowance)
    return({'N_i_j':thedata.iloc[:-1, 4:].fillna(0).transpose().values,
            'd_i':np.array(allowance),
            'c_j':np.ones(len(commodities)),
            'nbi':len(allowance),
            'nbj':len(commodities),
            'names_i': list(thedata.columns)[4:],
            'names_j':commodities}) 


def print_optimal_diet(q_j):
    print('***Optimal solution***')
    total,thelist = 0.0, []
    for j,commodity in enumerate(commodities):
        if q_j[j] > 0:
            total += q_j[j] * 365
            thelist.append([commodity,q_j[j]])
    thelist.append(['Total cost (optimal):', total])
    print(tabulate(thelist))


#########################
# LP2: The simplex algorithm #
#########################

def round_expr(expr, num_digits):
    return expr.xreplace({n : round(n, num_digits) for n in expr.atoms(Number)})

class Tableau():
    def __init__(self, names_basic, names_nonbasic, A_i_j, b_i, c_j): # z = d - A @ x
        self.A_i_j,self.b_i,self.c_j = A_i_j,b_i,c_j
        self.nonbasic = list(symbols(names_nonbasic))
        self.base = { Symbol('obj') : c_j @ self.nonbasic }
        self.base.update( { list(symbols(names_basic))[i]: b_i[i]  - (A_i_j @ self.nonbasic)[i] for i in range(len(b_i))} )

    def variables(self):
        return( list(self.base.keys())[1:] + self.nonbasic)

    def display(self):
        for i in self.base:
            print(i,' = ' ,round_expr(self.base[i],2))
            
    def solution(self, verbose=0):
        solution = {}
        for var in self.base:
            solution[var] = float(self.base[var].subs([(variable,0) for variable in self.nonbasic]))
            if verbose > 0: print(str(var) + ' = ' + str(solution[var]))
        for var in self.nonbasic:
            solution[var] = 0.0
            if verbose > 1: print(str(var) + ' = ' + str(solution[var]))
        return solution

    def determine_entering(self):
        self.nonbasic.sort(key=str) # Bland's rule
        for entering_var in self.nonbasic:
            if diff(self.base[Symbol('obj')],entering_var)>0 :
                return entering_var
        return None # If no entering variable found, None returned
    
    def determine_departing(self,entering_var):
        runmin = float('inf')
        departing_var = None
        for var in self.base.keys() - {Symbol('obj')}:
            the_expr_list = solve(self.base[var] - var,entering_var)
            if the_expr_list: # if one can invert the previous expression
                the_expr = the_expr_list[0] # express entering variable as a function of the other ones:
                val_entering_var = the_expr.subs([ (variable,0) for variable in [var]+self.nonbasic])
                if (val_entering_var >= 0) & (val_entering_var < runmin) :
                  runmin,departing_var = val_entering_var, var
        return departing_var # if no variable is found, None returned
        
    def pivot(self, entering_var, departing_var, verbose = 0):
        expr_entering = solve(self.base[departing_var] - departing_var, entering_var)[0]
        for var in self.base:
            self.base[var] = self.base[var].subs([(entering_var, expr_entering)])
        self.base[entering_var] = expr_entering
        del self.base[departing_var]
        self.nonbasic.remove(entering_var)
        self.nonbasic.append(departing_var)
        if verbose > 0:
            print('Entering = ' + str( entering_var)+'; departing = '+ str( departing_var))
        if verbose > 1:
            print(str( entering_var)+' = '+str(round_expr(expr_entering,2)))
        return expr_entering

    def simplex_loop(dual_tableau,primal_tableau,primals):
        entering_var = dual_tableau.determine_entering()
        if entering_var is None:
            print('Optimal solution found.\n=======================')
            primal_tableau.print_solution('Basic primal variables:')
            dual_tableau.print_solution('Basic dual variables:')
        else:
            departing_var = dual_tableau.determine_departing(entering_var)
            if departing_var is None:
                print('Unbounded solution.')
            else:
                expr_entering_var = dual_tableau.pivot(entering_var,departing_var, verbose= 1)
                _ = primal_tableau.pivot(primals[departing_var],primals[entering_var])
                return False # not finished
        return True # finished

    def plot_path_so_far(self, the_path, legend=True):
        if self.A_i_j.shape[1] != 2:
            print('Can\'t plot the solution in 2D, the number of primal variables differs from two.')
            return()

        x1max = min(x for x in self.b_i / self.A_i_j[:,0] if x >= 0)
        x2max = min(x for x in self.b_i / self.A_i_j[:,1] if x >= 0)
        x1, x2 = np.meshgrid(np.linspace(-.2*x1max, 1.4*x1max, 400), np.linspace(-.2*x2max, 1.4*x2max, 400))
        feasible_region = (x1 >= 0) & (x2 >= 0)
        for i in range(len(self.b_i)) :
            feasible_region = feasible_region & (self.A_i_j[i,0] * x1 + self.A_i_j[i,1] * x2 <= self.b_i[i])
        fig, ax = plt.subplots(figsize=(5, 5))
        plt.contourf(x1, x2, np.where(feasible_region, self.c_j[0]*x1+self.c_j[1]*x2, np.nan), 50, alpha = 0.5, cmap='gray_r', levels=30)
        for i in range(len(self.b_i)):
            if not self.A_i_j[i,1] == 0:
                ax.plot(x1[0, :], self.b_i[i]/self.A_i_j[i,1] - self.A_i_j[i,0]/self.A_i_j[i,1]*x1[0, :], label='z'+str(i+1)+' = 0')
            else:
                ax.axvline(self.b_i[i]/self.A_i_j[i,0], label='z'+str(i+1)+' = 0')
        ax.plot([a for (a,_) in the_path], [b for (_,b) in the_path], 'r--', label='Algorithm path')
        ax.scatter([a for (a,_) in the_path], [b for (_,b) in the_path], color='red')
        ax.set_xlim(-.2*x1max, 1.4*x1max), ax.set_ylim(-.2*x2max, 1.4*x2max)
        ax.set_xlabel('x1'), ax.set_ylabel('x2')
        ax.spines[ 'left' ].set_position('zero'), ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none'), ax.spines['top'].set_color('none')
        if legend: ax.legend(loc='upper right')
        plt.show()
        
#########################
# LP3: Interior Point Methods #
#########################

class InteriorPoint():
    def __init__(self,A,b,c,current_point=None):
        self.A = A
        self.b = b
        self.c = c
        self.current_point = current_point
        self.α = 1 - (1/8)/(1/5 + np.sqrt(len(c))) # shrinkage coeff α given by Freund & Vera

    def update(self, verbose=0):
        x, y, s, θ = self.current_point
        Δy = np.linalg.solve(self.A @ np.diag(1/s) @ np.diag(x) @ self.A.T, θ * self.A @ (1/s) - self.b)
        Δs = self.A.T @ Δy
        Δx = - x - np.diag(1/s) @ np.diag(x) @ Δs + θ * (1/s)
        self.current_point = [x+Δx, y+Δy, s+Δs, self.α*θ]
        return self.current_point
        
    def loop(self, tol=1e-6, max_iter=100, verbose=0):
        current_point = self.current_point
        new_point = IPM_update(self)
        if all(abs(np.concatenate(new_point[:-1]) - np.concatenate(current_point[:-1])) < tol):
            print('Optimal solution found.\n=======================')
            for i in range(len(new_point[0])): print("x_" + str(i+1), "=", new_point[0][i])
        else:
            if verbose > 1:
                for i in range(len(new_point[0])): print("x_" + str(i+1), "=", new_point[0][i])
            return False # not finished
        return True # finished
