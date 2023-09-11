import pandas as pd
import numpy as np
import scipy.sparse as spr
import gurobipy as grb
import sympy
from sympy.solvers import solve
from sympy import *
import matplotlib.pyplot as plt
from tabulate import tabulate


#############################
# LP1: Intro to linear programming #
#############################



def load_stigler_data(nbi = 9, nbj = 77, verbose=False):
    import pandas as pd
    thepath = 'https://raw.githubusercontent.com/math-econ-code/mec_optim_2021-01/master/data_mec_optim/lp_stigler-diet/'
    filename = 'StiglerData1939.txt'
    thedata = pd.read_csv(thepath + filename, sep='\t')
    thedata = thedata.dropna(how = 'all')
    commodities = (thedata['Commodity'].values)[:-1]
    allowance = thedata.iloc[-1, 4:].fillna(0).transpose()
    nbi = min(len(allowance),nbi)
    nbj = min(len(commodities),nbj)
    if verbose:
        print('Daily nutrient content:')
        print(tabulate(thedata.head()))
        print('\nDaily nutrient requirement:')
        print(allowance)
    return({'N_i_j':thedata.iloc[:nbj, 4:(4+nbi)].fillna(0).to_numpy().T,
            'd_i':np.array(allowance)[0:nbi],
            'c_j':np.ones(len(commodities))[0:nbj],
            'names_i': list(thedata.columns)[4:(4+nbi)],
            'names_j':commodities[0:nbj]}) 


def print_optimal_diet(q_j):
    print('***Optimal solution***')
    total,thelist = 0.0, []
    for j, commodity in enumerate(commodities):
        if q_j[j] > 0:
            total += q_j[j] * 365
            thelist.append([commodity,q_j[j]])
    thelist.append(['Total cost (optimal):', total])
    print(tabulate(thelist))

class LP():
    def __init__(self,A_i_j,d_i,c_j,var_names_j=None,slack_names_i=None):
        self.A_i_j = A_i_j
        self.nbi , self.nbj = A_i_j.shape
        self.nbk = self.nbi+self.nbj
        self.d_i = d_i
        self.c_j = c_j
        if var_names_j is None:
            var_names_j = ['x_'+str(j) for j in range(self.nbj)]
        if slack_names_i is None:
            slack_names_i = ['s_'+str(i) for i in range(self.nbi)]
        self.var_names_j = var_names_j
        self.slack_names_i = slack_names_i
        
    def gurobi_solve(self,verbose=0):
        import gurobipy as grb
        m = grb.Model()
        if verbose == 0:
            m.setParam('OutputFlag', 0)
        xg_j = m.addMVar(self.nbj)
        m.setObjective(xg_j@self.c_j,sense=grb.GRB.MAXIMIZE)
        constr_i = m.addConstr(self.A_i_j @ xg_j <= self.d_i)
        m.optimize()
        return(xg_j.x,constr_i.pi,m.objVal)
        
    
    def plot2d (self, the_path=[], legend=True):
        if len(self.c_j) != 2:
            print('The number of variables differs from two.')
            return()
        x1max = min(di/self.A_i_j[i,0] for i, di in enumerate(self.d_i) if self.A_i_j[i,0] != 0 and di/self.A_i_j[i,0] >= 0)
        x2max = min(di/self.A_i_j[i,1] for i, di in enumerate(self.d_i) if self.A_i_j[i,1] != 0 and di/self.A_i_j[i,1] >= 0)
        x1, x2 = np.meshgrid(np.linspace(-.2*x1max, 1.4*x1max, 400), np.linspace(-.2*x2max, 1.4*x2max, 400))
        feasible_region = (x1 >= 0) & (x2 >= 0)
        for i, di in enumerate(self.d_i):
            feasible_region = feasible_region & (self.A_i_j[i,0] * x1 + self.A_i_j[i,1] * x2 <= di)
        fig, ax = plt.subplots(figsize=(5, 5))
        plt.contourf(x1, x2, np.where(feasible_region, self.c_j[0]*x1 + self.c_j[1]*x2, np.nan), 50, alpha = 0.5, cmap='gray_r', levels=30)
        for i, di in enumerate(self.d_i):
            if self.A_i_j[i,1] != 0:
                ax.plot(x1[0, :], di/self.A_i_j[i,1] - self.A_i_j[i,0]/self.A_i_j[i,1]*x1[0, :], label=self.slack_names_i[i]+' = 0')
            else:
                ax.axvline(di/self.A_i_j[i,0], label=self.slack_names_i[i]+' = 0')
        if the_path:
            ax.plot([a for (a,_) in the_path], [b for (_,b) in the_path], 'r--', label='Agorithm path')
            ax.scatter([a for (a,_) in the_path], [b for (_,b) in the_path], color='red')
        ax.set_xlim(-.2*x1max, 1.4*x1max), ax.set_ylim(-.2*x2max, 1.4*x2max)
        ax.set_xlabel(self.var_names_j[0]), ax.set_ylabel(self.var_names_j[1])
        ax.spines[ 'left' ].set_position('zero'), ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none'), ax.spines['top'].set_color('none')
        if legend: ax.legend(loc='upper right')
        plt.show()


#########################
# LP2: The simplex algorithm #
#########################

def round_expr(expr, num_digits):
    return expr.xreplace({n : round(n, num_digits) for n in expr.atoms(Number)})

def limited_tabulate(data, headers=None, tablefmt='grid', max_rows=18, max_cols=14):
    if max_rows is not None and len(data) > max_rows:
        data = data[:max_rows]

    if max_cols is not None:
        if headers:
            headers = headers[:max_cols]
        data = [row[:max_cols] for row in data]
    
    return tabulate(data, headers=headers, tablefmt=tablefmt)

class Dictionary():
    def __init__(self, names_basic, names_nonbasic, A_i_j, d_i, c_j): # s = d - A @ x
        self.A_i_j, self.d_i, self.c_j = A_i_j, d_i, c_j
        self.init_names_basic = names_basic
        self.init_names_nonbasic = names_nonbasic
        self.nonbasic = list(symbols(names_nonbasic))
        self.base = { Symbol('obj') : c_j @ self.nonbasic }
        self.base.update( { list(symbols(names_basic))[i]: d_i[i]  - (A_i_j @ self.nonbasic)[i] for i in range(len(d_i))} )

    def variables(self):
        return( list(self.base.keys())[1:] + self.nonbasic)

    def display(self):
        print('-------------------------- \nObjective and constraints:')
        for var in self.base:
            print(var, '=', round_expr(self.base[var],2))
            
    def solution(self, verbose=0):
        solution = {}
        for var in self.base:
            solution[var] = float(self.base[var].subs([(variable,0) for variable in self.nonbasic]))
            if verbose > 0: print(var, '=', solution[var])
        for var in self.nonbasic:
            solution[var] = 0.0
            if verbose > 1: print(var, '=', solution[var])
        return solution

    # def plot_path (self, the_path, legend=True):
        # nbi,nbj = self.A_i_j.shape
        # if len(self.c_j[self.c_j!=0]) > 2:
            # print('Can\'t plot the solution in 2D: the vector self.c_j needs to have at most 2 nonzero entries.')
            # return()
        # x1max = min(di/self.A_i_j[i,0] for i, di in enumerate(self.d_i) if self.A_i_j[i,0] != 0 and di/self.A_i_j[i,0] >= 0)
        # x2max = min(di/self.A_i_j[i,1] for i, di in enumerate(self.d_i) if self.A_i_j[i,1] != 0 and di/self.A_i_j[i,1] >= 0)
        # x1, x2 = np.meshgrid(np.linspace(-.2*x1max, 1.4*x1max, 400), np.linspace(-.2*x2max, 1.4*x2max, 400))
        # feasible_region = (x1 >= 0) & (x2 >= 0)
        # for i, di in enumerate(self.d_i):
            # feasible_region = feasible_region & (self.A_i_j[i,0] * x1 + self.A_i_j[i,1] * x2 <= di)
        # fig, ax = plt.subplots(figsize=(5, 5))
        # plt.contourf(x1, x2, np.where(feasible_region, self.c_j[0]*x1 + self.c_j[1]*x2, np.nan), 50, alpha = 0.5, cmap='gray_r', levels=30)
        # for i, di in enumerate(self.d_i):
            # if self.A_i_j[i,1] != 0:
                # ax.plot(x1[0, :], di/self.A_i_j[i,1] - self.A_i_j[i,0]/self.A_i_j[i,1]*x1[0, :], label=self.init_names_basic[i]+' = 0')
            # else:
                # ax.axvline(di/self.A_i_j[i,0], label=self.init_names_basic[i]+' = 0')
        # if the_path:
            # ax.plot([a for (a,_) in the_path], [b for (_,b) in the_path], 'r--', label='Agorithm path')
            # ax.scatter([a for (a,_) in the_path], [b for (_,b) in the_path], color='red')
        # ax.set_xlim(-.2*x1max, 1.4*x1max), ax.set_ylim(-.2*x2max, 1.4*x2max)
        # ax.set_xlabel(self.init_names_nonbasic[0]), ax.set_ylabel(self.init_names_nonbasic[1])
        # ax.spines[ 'left' ].set_position('zero'), ax.spines['bottom'].set_position('zero')
        # ax.spines['right'].set_color('none'), ax.spines['top'].set_color('none')
        # if legend: ax.legend(loc='upper right')
        # plt.show()
    


    def determine_entering(self):
        self.nonbasic.sort(key=str) # Bland's rule
        for entering_var in self.nonbasic:
            if diff(self.base[Symbol('obj')],entering_var) > 0 :
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
        
    def pivot(self,entering_var,departing_var, verbose = 0):
        expr_entering = solve(self.base[departing_var] - departing_var,entering_var)[0]
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

    def loop(self):
        entering_var = self.determine_entering()
        if entering_var is None:
            print('Optimal solution found.\n=======================')
            self.solution(verbose=2)
        else:
            departing_var = self.determine_departing(entering_var)
            if departing_var is None:
                print('Unbounded solution.')
            else:
                expr_entering_var = self.pivot(entering_var,departing_var, verbose=1)
                return False # not finished
        return True # finished


#########################
# LP3: Interior Point Methods #
#########################

class InteriorPoint():
    def __init__(self, A, b, c, current_point=None):
        self.A, self.b, self.c = A, b, c
        self.current_point = current_point
        self.α = 1 - (1/8)/(1/5 + np.sqrt(len(self.c))) # shrinkage coeff from Freund & Vera

#    def strictly_feasible_solution(self):
#        x = np.linalg.lstsq(self.A, self.b) # Ax < b
#        s = .01*np.ones(len(self.c))
#        y = np.linalg.lstsq(self.A.T, s + self.c) # A.T y > c
#        return np.concatenate((x,y,s))

    def plot_path(self, the_path, legend=True):
        plot_path(self.A, self.b, self.c, the_path, legend)
    
    def update(self, verbose=0):
        x, y, s, θ = self.current_point
        Δy = np.linalg.solve(self.A @ np.diag(1/s) @ np.diag(x) @ self.A.T, θ * self.A @ (1/s) - self.b)
        Δs = self.A.T @ Δy
        Δx = - x - np.diag(1/s) @ np.diag(x) @ Δs + θ * (1/s)
        self.current_point = [x+Δx, y+Δy, s+Δs, self.α*θ]
        return self.current_point
    
    def IP_loop(self, tol=1e-6, verbose=0):
        current_point = self.current_point
        new_point = self.update()
        if all(abs(np.concatenate(new_point[:-1]) - np.concatenate(current_point[:-1])) < tol):
            print('Optimal solution found.\n=======================')
            if verbose > 0:
                for i in range(len(new_point[0])): print("x_" + str(i+1), "=", new_point[0][i])
            else:
                if verbose > 1:
                    for i in range(len(new_point[0])): print("x_" + str(i+1), "=", new_point[0][i])
                return False # not finished
        return True # finished


