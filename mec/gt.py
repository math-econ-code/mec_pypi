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
		alpha = model.addMVar(shape = 1, lb = -grb.GRB.INFINITY)
		beta = model.addMVar(shape = 1, lb = -grb.GRB.INFINITY)
		model.setObjective(p_i@(self.A_i_j+ self.B_i_j)@q_j - alpha - beta,sense = grb.GRB.MAXIMIZE )
		model.addConstr(self.A_i_j @ q_j - np.ones((self.nbi,1)) @  alpha <=  0 ) # 
		model.addConstr(self.B_i_j.T @ p_i <= np.ones((self.nbj,1)) @  beta ) # @ 
		model.addConstr(p_i.sum() == 1)
		model.addConstr(q_j.sum() == 1)
		model.optimize() 
		thesol = np.array( model.getAttr('x'))
		sol_dict = {'val1':thesol[-2], 'val2':thesol[-1], 'p_i':thesol[:self.nbi],'q_j':thesol[self.nbi:(self.nbi+self.nbj)]}    
		return(sol_dict)
