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





class TwoBases:
    def __init__(self,C_i_j,A_i_j,d_i = None):
        self.nbi,self.nbj = C_i_j.shape
        self.C_i_j = C_i_j
        self.A_i_j = A_i_j
        if d_i is  None:
            self.d_i = np.ones(self.nbi)
        else:
            self.d_i = d_i
        ###

    def is_standard_form(self):
        cond_1 = (np.diag(self.C_i_j)  == self.C_i_j.min(axis = 1) ).all() 
        cond_2 = ((self.C_i_j[:,:self.nbi] + np.diag([np.inf] * self.nbi)).min(axis=1) >= self.C_i_j[:,self.nbi:].max(axis=1)).all()
        return (cond_1 & cond_2)
    
    def remove_degeneracies(self, eps = 1e-5):
        self.d_i = self.d_i + np.arange(1,self.nbi+1)*eps
        
    def u_i(self,basis):
        return self.C_i_j[:,basis].min(axis = 1)
    
    def xsol_j(self,basis=None):
        if basis is None:
            basis = self.basis_C
        B = self.A_i_j[:,basis]
        x_j = np.zeros(self.nbj)
        x_j[basis] = np.linalg.solve(B,self.d_i)
        return x_j
    
    def is_feasible_basis(self,basis):    
        try:
            if self.xsol_j(basis).min()>=0:
                return True
        except np.linalg.LinAlgError:
            pass
        return False
        
    def is_ordinal_basis(self,basis):
        res=False
        if len(set(basis))==self.nbi:
            res = (self.u_i(basis)[:,None] >= self.C_i_j).any(axis = 0).all()
        return res

       
    def init_algo(self,j_removed):
        self.nbstep = 0
        self.tableau_A = Tableau( self.A_i_j[:,self.nbi:self.nbj], d_i = self.d_i, c_j = np.zeros(self.nbj-self.nbi), 
                        slack_var_names_i = [str(i) for i in range(self.nbi)],
                        decision_var_names_j =[str(i) for i in range(self.nbi,self.nbj)] )
        self.basis_C = list(range(self.nbi))
        self.basis_C.remove(j_removed)
        j_entering = self.nbi+self.C_i_j[j_removed,self.nbi:].argmax()
        self.basis_C.append(j_entering)
        return j_entering
    
        
    def continue_C(self,departing):
        ubefore_i = self.u_i(self.basis_C)
        self.basis_C.remove(departing)
        uafter_i = self.u_i(self.basis_C)
        i0 = np.where(ubefore_i < uafter_i)[0][0]
        c0 = min([(c,self.C_i_j[i0,c]) for c in self.basis_C  ],key = lambda x: x[1])[0]
        istar = [i for i in range(self.nbi) if uafter_i[i] == self.C_i_j[i,c0] and i != i0][0]
        eligible_columns = [c for c in range(self.nbj) if min( [self.C_i_j[i,c] - uafter_i[i] for i in range(self.nbi) if i != istar]) >0 ]
        cstar = max([(c,self.C_i_j[istar,c]) for c in eligible_columns], key = lambda x: x[1])[0]
        return cstar
        
    
    def step(self,entvar ,verbose= 0):
        self.nbstep += 1
        basis_A =  set(self.tableau_A.k_b)
        depvar = self.tableau_A.determine_departing(entvar)
        depcol = self.tableau_A.k_b[depvar]
        self.tableau_A.update(entvar,depvar)
        if set(self.basis_C) == set(self.tableau_A.k_b):
            if verbose>0:
                print('Step=', self.nbstep)
                print('Solution found. Basis=',set(self.basis_C) )
            return False
        ####
        u_i = self.u_i(self.basis_C)
        c0 = self.continue_C(depcol)
        ### 
        if verbose>0:
            print('Step=', self.nbstep)
            print('A basis = ' ,basis_A)
            print('C basis = ' ,set(self.basis_C))
            print('u_i=',u_i )
            print('entering var (A)=',entvar)
            print('departing var (A and C)=',depcol)
            print('entering var (C)=',c0)
        self.basis_C.append(c0)

        return c0