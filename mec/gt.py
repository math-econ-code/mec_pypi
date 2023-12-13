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
        tab2 = Dictionary( self.A_i_j, np.ones(self.nbi),np.zeros(self.nbj),ris, yjs )
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
    def __init__(self,Phi_z_a,M_z_a,q_z=None,remove_degeneracies=True,M=None,eps=1e-5):
        self.Phi_z_a,self.M_z_a = Phi_z_a,M_z_a
        if M is None:
            M = self.Phi_z_a.max()
        self.nbstep,self.M,self.eps = 1,M,eps
        self.nbz,self.nba = self.Phi_z_a.shape
        if q_z is  None:
            self.q_z = np.ones(self.nbz)
        else:
            self.q_z = q_z
        
        # remove degeneracies:
        if remove_degeneracies:
            self.Phi_z_a += np.arange(self.nba,0,-1)[None,:]* (self.Phi_z_a == self.M)
            self.q_z = self.q_z + np.arange(1,self.nbz+1)*self.eps
        # create an M and a Phi basis
        self.tableau_M = Tableau( self.M_z_a[:,self.nbz:self.nba], d_i = self.q_z )
        self.basis_Phi = list(range(self.nbz))
        ###
        
    def init_a_entering(self,a_removed):
        self.basis_Phi.remove(a_removed)
        a_entering = self.nbz+self.Phi_z_a[a_removed,self.nbz:].argmax()
        self.basis_Phi.append(a_entering)
        self.entvar = a_entering
        return a_entering
    
    def get_basis_M(self):
        return set(self.tableau_M.k_b)
    
    def get_basis_Phi(self):
        return set(self.basis_Phi)

    def is_standard_form(self):
        cond_1 = (np.diag(self.Phi_z_a)  == self.Phi_z_a.min(axis = 1) ).all() 
        cond_2 = ((self.Phi_z_a[:,:self.nbz] + np.diag([np.inf] * self.nbz)).min(axis=1) >= self.Phi_z_a[:,self.nbz:].max(axis=1)).all()
        return (cond_1 & cond_2)
    
        
    def p_z(self,basis=None):
        if basis is None:
            basis = self.get_basis_Phi()
        return self.Phi_z_a[:,list(basis)].min(axis = 1)    
    
    def musol_a(self,basis=None):
        if basis is None:
            basis = self.get_basis_M()
        B = self.M_z_a[:,list(basis)]
        mu_a = np.zeros(self.nba)
        mu_a[list(basis)] = np.linalg.solve(B,self.q_z)
        return mu_a

    
    def is_feasible_basis(self,basis):    
        try:
            if self.musol_a(list(basis) ).min()>=0:
                return True
        except np.linalg.LinAlgError:
            pass
        return False
        
    def is_ordinal_basis(self,basis):
        res, which =False,None
        if len(set(basis))==self.nbz:
            blocking = (self.Phi_z_a[:,basis].min(axis = 1)[:,None] < self.Phi_z_a).all(axis = 0)
            if blocking.any():
                which = np.where(blocking)
        return res, which
    
    
    def determine_entering(self,a_departing):
        self.nbstep += 1
        pbefore_z = self.p_z(self.basis_Phi)
        self.basis_Phi.remove(a_departing)
        pafter_z = self.p_z(self.basis_Phi)
        i0 = np.where(pbefore_z < pafter_z)[0][0]
        c0 = min([(c,self.Phi_z_a[i0,c]) for c in self.basis_Phi  ],key = lambda x: x[1])[0]
        zstar = [z for z in range(self.nbz) if pafter_z[z] == self.Phi_z_a[z,c0] and z != i0][0]
        eligible_columns = [c for c in range(self.nba) if min( [self.Phi_z_a[z,c] - pafter_z[z] for z in range(self.nbz) if z != zstar]) >0 ]
        a_entering = max([(c,self.Phi_z_a[zstar,c]) for c in eligible_columns], key = lambda x: x[1])[0]
        self.basis_Phi.append(a_entering)
        return a_entering
        
        
    
    def step(self,a_entering ,verbose= 0):
        a_departing = self.tableau_M.determine_departing(a_entering)
        self.tableau_M.update(a_entering,a_departing)
        
        if self.get_basis_M() ==self.get_basis_Phi():
            if verbose>0:
                print('Solution found in '+ str(self.nbstep)+' steps. Basis=',self.get_basis_Phi() )
            return False
            
        new_entcol = self.determine_entering(a_departing)

        if verbose>1:
            print('Step=', self.nbstep)
            print('M basis = ' ,self.get_basis_M() )
            print('Phi basis = ' ,self.get_basis_Phi() )
            print('p_z=',self.p_z(list(self.get_basis_Phi()) ))
            print('entering var (M)=',a_entering)
            print('departing var (M and Phi)=',a_departing)
            print('entering var (Phi)=',new_entcol)

        return new_entcol


    def solve(self,a_departing = 0, verbose=0):
        a_entering = self.init_a_entering(a_departing)
        while a_entering:
            a_entering = self.step(a_entering,verbose)
        return({'basis': self.get_basis_Phi(),
                'mu_a':self.musol_a(),
                'p_z':self.p_z()})
