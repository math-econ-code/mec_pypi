# Linear programming library

import pandas as pd
import numpy as np
import scipy.sparse as spr
import gurobipy as grb
import sympy
from sympy.solvers import solve
from sympy import *
import matplotlib.pyplot as plt
import tabulate as tb


def round_expr(expr, num_digits):
    return expr.xreplace({n : round(n, num_digits) for n in expr.atoms(Number)})

def limited_tabulate(data, headers=None, tablefmt='grid', max_rows=18, max_cols=14):
    if max_rows is not None and len(data) > max_rows:
        data = data[:max_rows]

    if max_cols is not None:
        if headers:
            headers = headers[:max_cols]
        data = [row[:max_cols] for row in data]

    return tb.tabulate(data, headers=headers, tablefmt=tablefmt)





class LP():
    def __init__(self, A_i_j, d_i, c_j = None, decision_var_names_j = None, slack_var_names_i = None):
        if c_j is None:
            c_j = np.zeros(A_i_j.shape[1])
        self.A_i_j = A_i_j
        self.nbi, self.nbj = A_i_j.shape
        self.nbk = self.nbi+self.nbj
        self.d_i = d_i
        self.c_j = c_j
        if decision_var_names_j is None:
            decision_var_names_j = ['x_'+str(j) for j in range(self.nbj)]
        if slack_var_names_i is None:
            slack_var_names_i = ['s_'+str(i) for i in range(self.nbi)]
        self.decision_var_names_j = decision_var_names_j
        self.slack_var_names_i = slack_var_names_i

    def gurobi_solve(self,verbose=0):
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
                ax.plot(x1[0, :], di/self.A_i_j[i,1] - self.A_i_j[i,0]/self.A_i_j[i,1]*x1[0, :], label=self.slack_var_names_i[i]+' = 0')
            else:
                ax.axvline(di/self.A_i_j[i,0], label=self.slack_var_names_i[i]+' = 0')
        if the_path:
            ax.plot([a for (a,_) in the_path], [b for (_,b) in the_path], 'r--', label='Agorithm path')
            ax.scatter([a for (a,_) in the_path], [b for (_,b) in the_path], color='red')
        ax.set_xlim(-.2*x1max, 1.4*x1max), ax.set_ylim(-.2*x2max, 1.4*x2max)
        ax.set_xlabel(self.decision_var_names_j[0]), ax.set_ylabel(self.decision_var_names_j[1])
        ax.spines[ 'left' ].set_position('zero'), ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none'), ax.spines['top'].set_color('none')
        if legend: ax.legend(loc='upper right')
        plt.show()



class Dictionary(LP):
    def __init__(self, A_i_j, d_i, c_j = None, slack_var_names_i=None,decision_var_names_j=None):
        # s_i = d_i - A_i_j @ x_j
        if d_i.min()<0:
            from warnings import warn
            warn('The array d_i has negative entries; zero is not a feasible solution.')
        LP.__init__(self, A_i_j, d_i, c_j, decision_var_names_j, slack_var_names_i)
        self.nonbasic = [Symbol(x) for x in self.decision_var_names_j]
        self.base = { Symbol('obj') : c_j @ self.nonbasic }
        slack_exprs_i = d_i  - A_i_j @ self.nonbasic
        self.base.update({Symbol(name): slack_exprs_i[i] for (i,name) in enumerate(self.slack_var_names_i) })

    def variables(self):
        return( list(self.base.keys())[1:] + self.nonbasic)

    def display(self):
        print('-------------------------- \nObjective and constraints:')
        for var in self.base:
            print(var, '=', round_expr(self.base[var],2))


    def primal_solution(self, verbose=0):
        x_j = np.zeros(self.nbj)
        for j,var in enumerate([Symbol(x) for x in self.decision_var_names_j]):
            x_j[j]=float( self.base.get(var,sympy.Integer(0)).subs([(variable,0) for variable in self.nonbasic]) )
            if verbose > 0:
                print(var, '=', x_j[j])
        return x_j

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

    def step(self,verbose=0):
        entering_var = self.determine_entering()
        if entering_var is None:
            print('Optimal solution found.\n=======================')
            self.primal_solution(verbose)
        else:
            departing_var = self.determine_departing(entering_var)
            if departing_var is None:
                print('Unbounded solution.')
            else:
                expr_entering_var = self.pivot(entering_var,departing_var, verbose)
                return False # not finished
        return True # finished

    def dual_solution(self,verbose = 0):
        y_i = np.zeros(self.nbi)
        for i,slackvar in enumerate(self.slack_var_names_i):
            y_i[i] = - diff(self.base[Symbol('obj')],slackvar)
            if verbose > 0 and y_i[i] != 0:
                print('pi_'+str(i)+'=', y_i[i])
        return y_i


    def simplex_loop(self,verbose = 0):
        if self.d_i.min()<0:
            from warnings import warn
            warn('The array d_i has negative entries; zero is not a feasible solution.')
        if verbose >2:
            [x1,x2] = [Symbol(x) for x in self.decision_var_names_j]
            the_path = [self.primal_solution()]
        finished = False
        while not finished:
            finished = self.step()
            if verbose>2:
                the_path.append(self.primal_solution())
        objVal = self.base[Symbol('obj')].subs([ (variable,0) for variable in self.nonbasic])
        if verbose>0:
            print('\nValue = ' + str(objVal))
        if verbose >2:
            self.plot2d(the_path, legend=False)
        return (self.primal_solution(),self.dual_solution(),objVal)


class Tableau(LP):
    def __init__(self, A_i_j, d_i, c_j = None, slack_var_names_i = None, decision_var_names_j = None): # A_i_j @ x_j + s_i = d_i
        LP.__init__(self, A_i_j, d_i, c_j, decision_var_names_j, slack_var_names_i)
        self.nbi,self.nbj = A_i_j.shape
        self.nbk = self.nbi + self.nbj
        if c_j is None:
            c_j = np.zeros(self.nbj)
        if decision_var_names_j is None:
            decision_var_names_j = ['x_'+str(j) for j in range(self.nbj)]
        if slack_var_names_i is None:
            slack_var_names_i = ['s_'+str(i) for i in range(self.nbi)]
        self.names_all_variables =  self.slack_var_names_i + self.decision_var_names_j
        self.tableau = np.block( [[np.zeros((1,self.nbi)), c_j.reshape((1,-1)), 0],
                                  [np.eye(self.nbi), A_i_j, d_i.reshape((-1,1))]] )
        self.k_b = list(range(self.nbi)) # columns associated with basic variables
        self.i_b = list(range(1,1+self.nbi)) # rows associated with basic variables

    def display(self):
        tableau = []
        tableau.append( ['Obj'] + list(self.tableau[0,:]) )
        for b in range(self.nbi):
            tableau.append([self.names_all_variables[self.k_b[b]]]+list(self.tableau[self.i_b[b],:]) )
        print(limited_tabulate(tableau, headers=[''] + self.names_all_variables + ['RHS'], tablefmt="grid"))

    def determine_entering(self):
        for k in range(self.nbk):
            if self.tableau[0,k] > 0:
                return k
        return None # If no entering variable found, None returned

    #def determine_departing(self, kent): # Alfred
    #    thedic = {self.k_b[b]: self.tableau[self.i_b[b],-1] / self.tableau[self.i_b[b],kent]
    #              for b in range(self.nbi) if self.tableau[self.i_b[b],kent]>0}
    #    kdep = min(thedic, key = thedic.get)
    #    return kdep

    def determine_departing(self, kent):
        runmin, kdep = float('inf'), None
        for b in range(self.nbi):
            if self.tableau[self.i_b[b],kent] > 0:
                ratio = self.tableau[self.i_b[b],-1] / self.tableau[self.i_b[b],kent]
                if (ratio < runmin):
                    runmin, kdep = ratio, self.k_b[b]
        return kdep

    def update(self, kent, kdep):
        bdep = self.k_b.index(kdep)
        idep = self.i_b[bdep]
        self.tableau[idep,:] = self.tableau[idep,:] / self.tableau[idep,kent]
        for i in range(1+self.nbi):
            if i != idep:
                self.tableau[i,:]= self.tableau[i,:] - self.tableau[idep,:] * self.tableau[i,kent]
        self.k_b[bdep] = kent
        self.i_b[bdep] = idep

    def simplex_step(self,verbose=0):
        if verbose>1:
            self.display()
        kent = self.determine_entering()
        if kent is not None:
            kdep= self.determine_departing(kent)
            if verbose>0:
                bdep = int(np.where(self.k_b == kdep)[0])
                print('Entering=', self.names_all_variables[kent], 'Departing=',self.names_all_variables[self.i_b[bdep]],'Pivot=',(self.i_b[bdep],kent))
            self.update(kent,kdep)
        else:
            if verbose>0:
                print ('Optimal solution found.')
            if verbose>1:
                self.display()
        return (kent is not None) # returns false  if optimal solution; true otherwise

    def simplex_solve(self,verbose=0):
        if self.d_i.min()<0:
            from warnings import warn
            warn('The array d_i has negative entries; zero is not a feasible solution.')
        while self.simplex_step(verbose):
            pass
        return self.solution()

    def solution(self): # returns the primal variables x_j and dual variables y_i
        x_j, s_i = np.zeros(self.nbj), np.zeros(self.nbi)
        for b in range(self.nbi):
            if self.k_b[b]<self.nbi:
                s_i[self.k_b[b]] = self.tableau[self.i_b[b],-1]
            else:
                x_j[self.k_b[b]-self.nbi] = self.tableau[self.i_b[b],-1]
        y_i = - self.tableau[0,:self.nbi]
        return x_j, y_i, x_j@self.c_j

##########################################
######### Interior Point Methods #########
##########################################

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


def two_phase(A_i_j,d_i, verbose = False):
    nbi,nbj = A_i_j.shape
    signs_i = np.minimum(2*np.sign(d_i)+1,1) # 1 if >=0, -1 else
    d_i = signs_i*d_i
    A_i_j = signs_i[:,None] * A_i_j
    the_tableau = Tableau(A_i_j, d_i, c_j = A_i_j.sum(axis= 0) )
    the_tableau.simplex_solve()
    if (min(the_tableau.k_b) >= nbi ):
        if verbose:
            print('Feasible.')
        return [k-nbi for k in the_tableau.k_b ]
    else:
        if verbose:
            print('Infeasible.')
        return None

    
