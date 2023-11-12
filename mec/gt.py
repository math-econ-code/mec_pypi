import gurobipy as grb
import numpy as np
from mec.lp import Dictionary
from mec.lp import Tableau



class Matrix_game:
    def __init__(self,Phi_i_j):
        self.nbi,self.nbj = Phi_i_j.shape
        self.Phi_i_j = Phi_i_j

    def BRI(self,j):
        return np.argwhere(self.Phi_i_j[:,j] == np.max(self.Phi_i_j[:,j])).flatten()

    def BRJ(self,i):
        return np.argwhere(self.Phi_i_j[i,:] == np.min(self.Phi_i_j[i,:])).flatten()

    def compute_eq(self):
        return [ (i,j) for i in range(self.nbi) for j in range(self.nbj) if ( (i in self.BRI(j) ) and (j in self.BRJ(i) ) ) ]

    def minimax_LP(self):
        model=grb.Model()
        model.Params.OutputFlag = 0
        y = model.addMVar(shape=self.nbj)
        model.setObjective(np.ones(self.nbj) @ y, grb.GRB.MAXIMIZE)
        model.addConstr(self.Phi_i_j @ y <= np.ones(self.nbi))
        model.optimize() 
        ystar = np.array(model.getAttr('x'))
        xstar = np.array(model.getAttr('pi'))
        S = 1 /  xstar.sum()
        p_i = S * xstar
        q_j = S * ystar
        return(p_i,q_j)
        
    def minimax_CP(self,gap_threshold = 1e-5,max_iter = 10000):
        L1 = np.max(np.abs(self.Phi_i_j))
        sigma, tau = 1/L1, 1/L1

        p = np.ones(self.nbi) / self.nbi
        q = np.ones(self.nbi) / self.nbj
        q_prev = q.copy()

        gap = np.inf
        i=0
        while (gap >  gap_threshold) and (i < max_iter):
            q_tilde = 2*q - q_prev
            p *= np.exp(-sigma* self.Phi_i_j @ q_tilde)
            p /= p.sum()

            q_prev = q.copy()
            q *= np.exp(tau* self.Phi_i_j.T @ p)
            q /= q.sum()
            gap = np.max(self.Phi_i_j.T@p) - np.min(self.Phi_i_j@q)
            i += 1
        return(p,q,gap,i)


class Bimatrix_game:
    def __init__(self,A_i_j,B_i_j):
        self.A_i_j = A_i_j
        self.B_i_j = B_i_j
        self.nbi,self.nbj = A_i_j.shape

    def mangasarian_stone_solve(self):
        model=grb.Model()
        model.Params.OutputFlag = 0
        model.params.NonConvex = 2
        p_i = model.addMVar(shape=self.nbi)
        q_j = model.addMVar(shape=self.nbj)
        alpha = model.addMVar(shape = 1)
        beta = model.addMVar(shape = 1)
        model.setObjective(alpha + beta  - p_i@(self.A_i_j+ self.B_i_j)@q_j ,sense = grb.GRB.MINIMIZE )
        model.addConstr(self.A_i_j @ q_j - np.ones((self.nbi,1)) @  alpha <=  0 ) # 
        model.addConstr(self.B_i_j.T @ p_i <= np.ones((self.nbj,1)) @  beta ) # @ 
        model.addConstr(p_i.sum() == 1)
        model.addConstr(q_j.sum() == 1)
        model.optimize() 
        thesol = np.array( model.getAttr('x'))
        sol_dict = {'val1':thesol[-2], 'val2':thesol[-1], 'p_i':thesol[:self.nbi],'q_j':thesol[self.nbi:(self.nbi+self.nbj)]}    
        return(sol_dict)
        
    def lemke_howson_solve(self,verbose = 0):
        from sympy import Symbol
        
        ris = ['r_' + str(i+1) for i in range(self.nbi)]
        yjs = ['y_' + str(self.nbi+j+1) for j in range(self.nbj)]
        sjs = ['s_' + str(self.nbi+j+1) for j in range(self.nbj)]
        xis = ['x_' + str(i+1) for i in range(self.nbi)]
        #tab2 = Tableau(ris, yjs, self.A_i_j, np.ones(self.nbi) )
        tab2 = Dictionary( self.A_i_j, np.ones(self.nbi),np.zeros(self.nbi),ris, yjs )
        #tab1 = Tableau(sjs, xis, self.B_i_j.T, np.ones(self.nbj) )
        tab1 = Dictionary(self.B_i_j.T, np.ones(self.nbj), np.zeros(self.nbi), sjs, xis)
        keys = ris+yjs+sjs+xis
        labels = xis+sjs+yjs+ris
        complements = {Symbol(keys[t]): Symbol(labels[t]) for t in range(len(keys))}
        entering_var1 = Symbol('x_1')
            
        while True:
            if not (entering_var1 in set(tab1.nonbasic)):
                #print('Equilibrium found (1).')
                break
            departing_var1 = tab1.determine_departing(entering_var1)
            tab1.pivot(entering_var1,departing_var1,verbose=verbose)
            entering_var2 = complements[departing_var1]
            if not (entering_var2 in set(tab2.nonbasic)):
                #print('Equilibrium found (2).')
                break
            else:
                departing_var2 = tab2.determine_departing(entering_var2)
                tab2.pivot(entering_var2,departing_var2,verbose=verbose)
                entering_var1 = complements[departing_var2]
        x_i = tab1.primal_solution()
        y_j = tab2.primal_solution()
        
        val1 = 1 / y_j.sum()
        val2 = 1 /  x_i.sum()
        p_i = x_i * val2
        q_j = y_j * val1
        sol_dict = {'val1':val1, 'val2':val2, 'p_i':p_i,'q_j':q_j}
        return(sol_dict)

import numpy as np
from mec.lp import Tableau

class TwoBases:
    def __init__(self,C_i_j,A_i_j,d_i=None,M=None,eps=1e-5):
        if M is None:
            M = C_i_j.max()
        self.nbstep,self.M,self.eps = 1,M,eps
        self.nbi,self.nbj = C_i_j.shape
        self.C_i_j = C_i_j
        self.A_i_j = A_i_j
        if d_i is  None:
            self.d_i = np.ones(self.nbi)
        else:
            self.d_i = d_i
        # remove degeneracies:
        self.C_i_j += np.arange(self.nbj,0,-1)[None,:]* (self.C_i_j == self.M)
        self.d_i = self.d_i + np.arange(1,self.nbi+1)*self.eps
        # create an A and a C basis
        self.tableau_A = Tableau( self.A_i_j[:,self.nbi:self.nbj], d_i = self.d_i )
        self.basis_C = list(range(self.nbi))
        ###
        
    def init_j_entering(self,j_removed):
        self.basis_C.remove(j_removed)
        j_entering = self.nbi+self.C_i_j[j_removed,self.nbi:].argmax()
        self.basis_C.append(j_entering)
        self.entvar = j_entering
        return j_entering
    
    def get_basis_A(self):
        return set(self.tableau_A.k_b)
    
    def get_basis_C(self):
        return set(scarf_example.basis_C)

    def is_standard_form(self):
        cond_1 = (np.diag(self.C_i_j)  == self.C_i_j.min(axis = 1) ).all() 
        cond_2 = ((self.C_i_j[:,:self.nbi] + np.diag([np.inf] * self.nbi)).min(axis=1) >= self.C_i_j[:,self.nbi:].max(axis=1)).all()
        return (cond_1 & cond_2)
    
        
    def u_i(self,basis=None):
        if basis is None:
            basis = self.get_basis_C()
        return self.C_i_j[:,list(basis)].min(axis = 1)    
    
    def xsol_j(self,basis=None):
        if basis is None:
            basis = self.get_basis_A()
        B = self.A_i_j[:,list(basis)]
        x_j = np.zeros(self.nbj)
        x_j[list(basis)] = np.linalg.solve(B,self.d_i)
        return x_j

    
    def is_feasible_basis(self,basis):    
        try:
            if self.xsol_j(list(basis) ).min()>=0:
                return True
        except np.linalg.LinAlgError:
            pass
        return False
        
    def is_ordinal_basis(self,basis):
        res=False
        if len(set(basis))==self.nbi:
            res = (self.u_i(list(basis) )[:,None] >= self.C_i_j).any(axis = 0).all()
        return res
    
    
    def determine_entering(self,depcol):
        self.nbstep += 1
        ubefore_i = self.u_i(self.basis_C)
        self.basis_C.remove(depcol)
        uafter_i = self.u_i(self.basis_C)
        i0 = np.where(ubefore_i < uafter_i)[0][0]
        c0 = min([(c,self.C_i_j[i0,c]) for c in self.basis_C  ],key = lambda x: x[1])[0]
        istar = [i for i in range(self.nbi) if uafter_i[i] == self.C_i_j[i,c0] and i != i0][0]
        eligible_columns = [c for c in range(self.nbj) if min( [self.C_i_j[i,c] - uafter_i[i] for i in range(self.nbi) if i != istar]) >0 ]
        entcol = max([(c,self.C_i_j[istar,c]) for c in eligible_columns], key = lambda x: x[1])[0]
        self.basis_C.append(entcol)
        return entcol
        
        
    
    def step(self,entcol ,verbose= 0):
        depcol = self.tableau_A.determine_departing(entcol)
        self.tableau_A.update(entcol,depcol)
        
        if self.get_basis_A() ==self.get_basis_C():
            if verbose>0:
                print('Solution found in '+ str(self.nbstep)+' steps. Basis=',self.get_basis_C() )
            return False
            
        new_entcol = self.determine_entering(depcol)

        if verbose>1:
            print('Step=', self.nbstep)
            print('A basis = ' ,self.get_basis_A() )
            print('C basis = ' ,self.get_basis_C() )
            print('u_i=',self.u_i(list(self.get_basis_C()) ))
            print('entering var (A)=',entcol)
            print('departing var (A and C)=',depcol)
            print('entering var (C)=',new_entcol)

        return new_entcol


    def solve(self,depcol = 0, verbose=0):
        entcol = self.init_j_entering(depcol)
        while entcol:
            entcol = self.step(entcol,verbose)
        return({'basis': self.get_basis_C(),
                'x_j':self.xsol_j(),
                'u_i':self.u_i()})

