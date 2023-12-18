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


class LCP: # z >= 0, w = M z + q >= 0, z.w = 0
    def __init__(self,M_i_j,q_i):
        self.M_i_j = M_i_j
        self.nbi,_ = M_i_j.shape
        self.q_i = q_i

    def qp_solve(self, silent = True, verbose = 0):
        qp = grb.Model()
        if silent:
            qp.Params.OutputFlag = 0
        qp.Params.NonConvex = 2
        zqp_i = qp.addMVar(shape = self.nbi)
        wqp_i = qp.addMVar(shape = self.nbi)
        qp.addConstr(self.M_i_j @ zqp_i - wqp_i == - self.q_i)
        qp.setObjective(zqp_i @ wqp_i, sense = grb.GRB.MINIMIZE)
        qp.optimize()
        z_i = np.array(qp.getAttr('x'))[:self.nbi]
        if verbose > 0: print(z_i)
        if verbose > 1:
            w_i = self.M_i_j @ z_i + self.q_i
            print(w_i)
            print(z_i @ w_i)
        return(z_i)

    def plot_cones(self):
        if self.nbi != 2:
            raise ValueError('Can\'t plot in 2D because the problem is of order different from 2.')
        A, q = -self.M_i_j, self.q_i
        I = np.eye(2)
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.spines['left'].set_position('zero'), ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none'), ax.spines['top'].set_color('none')
        r = np.array([[.6,.7],[.8,.9]])
        angles = np.array([[0, np.pi/2], [None, None]])
        for j in range(2):
            if any(A[:,j] != 0): angles[1,j] = (-1)**(A[1,j]<0) * np.arccos(A[0,j]/np.linalg.norm(A[:,j]))
        for i in range(2):
            for j in range(2):
                angle_1 = angles[i,0]
                angle_2 = angles[j,1]
                if angle_1 != None and angle_2 != None:
                    if abs(angle_1 - angle_2) < np.pi:
                        theta = np.linspace(np.minimum(angle_1, angle_2), np.maximum(angle_1, angle_2), 100)
                    elif abs(angle_1 - angle_2) > np.pi:
                        theta = np.linspace(np.maximum(angle_1, angle_2), np.minimum(angle_1, angle_2) + 2*np.pi, 100)
                    ax.fill(np.append(0, r[i,j]*np.cos(theta)), np.append(0, r[i,j]*np.sin(theta)), 'b', alpha=0.1)
        for i in range(2):
            ax.quiver(0, 0, I[0,i], I[1,i], scale=1, scale_units='xy', color='b')
            ax.text(I[0,i], I[1,i], r'$E_{{{}}}$'.format(i+1), fontsize=12, ha='left', va='bottom')
            if any(A[:,i] != 0):
                ax.quiver(0, 0, A[0,i]/np.linalg.norm(A[:,i]), A[1,i]/np.linalg.norm(A[:,i]), scale=1, scale_units='xy', color='b')
                ax.text(A[0,i]/np.linalg.norm(A[:,i]), A[1,i]/np.linalg.norm(A[:,i]), r'$-M_{{\cdot{}}}$'.format(i+1), fontsize=12, ha='left', va='bottom')
        ax.quiver(0, 0, q[0]/np.linalg.norm(q), q[1]/np.linalg.norm(q), angles='xy', scale_units='xy', scale=1, color='r')
        plt.xlim(-1.2, 1.2), plt.ylim(-1.2, 1.2)
        plt.show()

    def lemke_solve(self,verbose=0,maxit=100):
        counter = 0
        zis = ['z_' + str(i+1) for i in range(self.nbi)]+['z_0']
        wis = ['w_' + str(i+1) for i in range(self.nbi)]
        if all(self.q_i >= 0):
            print('==========')
            print('Solution found: trivial LCP (q >= 0).')
            zsol=np.zeros(self.nbi)
            wsol=self.q_i
            for i in range(self.nbi): print(zis[i] + ' = 0')
            for i in range(self.nbi): print(wis[i] + ' = ' + str(wsol[i]))
            return zsol,wsol
        else:
            keys = wis + zis[:-1]
            labels = zis[:-1] + wis
            complements = {Symbol(keys[i]): Symbol(labels[i]) for i in range(2*self.nbi)}
            tab = Tableau(wis, zis, - np.block([self.M_i_j, np.ones((self.nbi,1))]), self.q_i)
            entering_var = Symbol('z_0')
            departing_var = Symbol(wis[np.argmin(self.q_i)]) # if argmin not unique, takes smallest index (Bland's rule)
            while departing_var is not None and counter < maxit:
                counter += 1
                tab.pivot(entering_var,departing_var,verbose=verbose)
                if departing_var == Symbol('z_0') or counter == maxit:
                    break
                else:
                    entering_var = complements[departing_var]
                    departing_var = tab.determine_departing(entering_var)
            print('==========')
            if departing_var == Symbol('z_0'):
                print('Solution found: z_0 departed basis.')
                sol=tab.solution()
                zsol=np.array([sol[Symbol(zis[i])] for i in range(self.nbi)])
                wsol=np.array([sol[Symbol(wis[i])] for i in range(self.nbi)])
                print('Complementarity: z.w = ' + str(zsol @ wsol))
                for i in range(2*self.nbi): print(labels[i] + ' = ' + str(sol[Symbol(labels[i])]))
                return zsol, wsol
            elif counter == maxit:
                print('Solution not found: maximum number of iterations (' + str(maxit) + ') reached.')
                return None
            else:
                print('Solution not found: Ray termination.')
                return None


class Bimatrix_game:
    def __init__(self, A_i_j, B_i_j):
        if A_i_j.shape != B_i_j.shape:
            raise ValueError("A_i_j and B_i_j must be of the same size.")
        self.A_i_j = A_i_j
        self.B_i_j = B_i_j
        self.nbi,self.nbj = A_i_j.shape

    def mangasarian_stone_solve(self):
        model=grb.Model()
        model.Params.OutputFlag = 0
        model.params.NonConvex = 2
        p_i = model.addMVar(shape = self.nbi)
        q_j = model.addMVar(shape = self.nbj)
        α = model.addMVar(shape = 1, lb = -grb.GRB.INFINITY)  # alternative to lb=-inf: add constant to ensure A_i_j > 0
        β = model.addMVar(shape = 1, lb = -grb.GRB.INFINITY)  # and B_i_j > 0, and adjust val1 and val2 at the end
        model.setObjective(α + β - p_i @ (self.A_i_j + self.B_i_j) @ q_j, sense = grb.GRB.MINIMIZE)
        model.addConstr(α * np.ones((self.nbi,1)) - self.A_i_j @ q_j >= 0)
        model.addConstr(β * np.ones((self.nbj,1)) - self.B_i_j.T @ p_i >= 0)
        model.addConstr(p_i.sum() == 1)
        model.addConstr(q_j.sum() == 1)
        model.optimize()
        thesol = np.array(model.getAttr('x'))
        sol_dict = {'val1': thesol[-2], 'val2': thesol[-1],
                    'p_i': thesol[:self.nbi], 'q_j': thesol[self.nbi:(self.nbi+self.nbj)]}
        return(sol_dict)

    def lemke_howson_solve(self,verbose = 0):
        A_i_j = self.A_i_j - np.min(self.A_i_j) + 1     # ensures that matrices are positive
        B_i_j = self.B_i_j - np.min(self.B_i_j) + 1
        zks = ['x_' + str(i+1) for i in range(self.nbi)] + ['y_' + str(j+1) for j in range(self.nbj)]
        wks = ['r_' + str(i+1) for i in range(self.nbi)] + ['s_' + str(j+1) for j in range(self.nbj)]
        complements = list(len(zks)+np.arange(len(zks))) + list(np.arange(len(zks)))
        C_k_l = np.block([[np.zeros((self.nbi, self.nbi)), A_i_j],
                          [B_i_j.T, np.zeros((self.nbj, self.nbj))]])
        tab = Tableau(C_k_l, np.ones(self.nbi + self.nbj), np.zeros(self.nbi + self.nbj), wks, zks)
        kent = len(wks) # z_1 enters
        while True:
            kdep = tab.determine_departing(kent)
            if verbose > 1:
                print('Basis: ', [(wks+zks)[i] for i in tab.k_b])
                print((wks+zks)[kent], 'enters,', (wks+zks)[kdep], 'departs')
            tab.update(kent, kdep)
            if (complements[kent] not in tab.k_b) and (complements[kdep] in tab.k_b):
                break
            else:
                kent = complements[kdep]
        z_k, _, _ = tab.solution() # solution returns: x_j, y_i, x_j@self.c_j
        x_i, y_j = z_k[:self.nbi], z_k[self.nbi:]
        α = 1 / y_j.sum()
        β = 1 /  x_i.sum()
        p_i = x_i * β
        q_j = y_j * α
        sol_dict = {'val1': α + np.min(self.A_i_j) - 1,
                    'val2': β + np.min(self.B_i_j) - 1,
                    'p_i': p_i, 'q_j': q_j}
        return(sol_dict)


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
