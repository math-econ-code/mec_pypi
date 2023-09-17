import gurobipy as grb
import numpy as np
from mec.lp import Dictionary


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
