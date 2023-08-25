import gurobipy as grb
import numpy as np


class Matrix_game:
    def __init__(self,Phi_i_j):
        self.nbi,self.nbj = Phi_i_j.shape
        self.Phi_i_j = Phi_i_j


	def BRI(self,j):
		return np.argwhere(self.Phi_i_j[:,j] == np.max(self.Phi_i_j[:,j])).flatten()

	def BRJ(self,i):
		return np.argwhere(self.Phi_i_j[i,:] == np.min(self.Phi_i_j[i,:])).flatten()

	def compute_eq(self):
		return [ (i,j) for i in range(self.nbi) for j in range(self.nbj)
				if ( (i in self.BRI(j) ) and (j in self.BRJ(i) ) ) 
				
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