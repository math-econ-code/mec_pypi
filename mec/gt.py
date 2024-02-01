import gurobipy as grb
import numpy as np
import sympy
from sympy import *
from mec.lp import Dictionary
from mec.lp import Tableau


class Matrix_game:
    def __init__(self, A_i_j):
        self.nbi,self.nbj = A_i_j.shape
        self.A_i_j = A_i_j

    def BRI(self,j):
        return np.argwhere(self.A_i_j[:,j] == np.max(self.A_i_j[:,j])).flatten()

    def BRJ(self,i):
        return np.argwhere(self.A_i_j[i,:] == np.min(self.A_i_j[i,:])).flatten()

    def purestrat_solve(self):
        return [ (i,j) for j in range(self.nbj) for i in range(self.nbi) if ( (i in self.BRI(j)) and (j in self.BRJ(i)) ) ]

    def solve(self, verbose=0):
        model = grb.Model()
        model.Params.OutputFlag = 0
        y = model.addMVar(shape=self.nbj)
        model.setObjective(np.ones(self.nbj) @ y, grb.GRB.MAXIMIZE)
        model.addConstr(self.A_i_j @ y <= np.ones(self.nbi))
        model.optimize()
        ystar = np.array(model.getAttr('x'))
        xstar = np.array(model.getAttr('pi'))
        V_2 = 1 / model.getAttr('ObjVal')
        p_i = V_2 * xstar
        q_j = V_2 * ystar
        if verbose > 0: print('p_i =', p_i, '\nq_j =', q_j)
        return {'p_i': p_i, 'q_j': q_j, 'val': V_2}

    def simplex_solve(self, verbose=0):
        tableau = Tableau(A_i_j = self.A_i_j, d_i = np.ones(self.nbi), c_j = np.ones(self.nbj),
                          decision_var_names_j = ['y_'+str(j) for j in range(self.nbj)])
        ystar, xstar, ystar_sum = tableau.simplex_solve()
        p_i = xstar / ystar_sum
        q_j = ystar / ystar_sum
        if verbose > 0: print('p_i =', p_i, '\nq_j =', q_j)
        return {'p_i': p_i, 'q_j': q_j, 'val': 1/ystar_sum}

    def chambolle_pock_solve(self, tol=10e-6, max_iter=10000):
        L1 = np.max(np.abs(self.A_i_j))
        sigma, tau = 1/L1, 1/L1

        p_i = np.ones(self.nbi) / self.nbi
        q_j = np.ones(self.nbi) / self.nbj
        q_prev = q_j.copy()

        gap = np.inf
        i=0
        while (gap >  tol) and (i < max_iter):
            q_tilde = 2*q_j - q_prev
            p_i *= np.exp(-sigma * self.A_i_j @ q_tilde)
            p_i /= p_i.sum()

            q_prev = q_j.copy()
            q_j *= np.exp(tau * self.A_i_j.T @ p_i)
            q_j /= q_j.sum()
            gap = np.max(self.A_i_j.T @ p_i) - np.min(self.A_i_j @ q_j)
            i += 1
        return p_i, q_j, gap, i


class LCP: # z >= 0, w = M z + q >= 0, z.w = 0
    def __init__(self, M_i_j, q_i, z_names_i = None, w_names_i = None):
        if M_i_j.shape[0] != M_i_j.shape[1]:
            raise ValueError("M_i_j must be square.")
        if M_i_j.shape[0] != len(q_i):
            raise ValueError("M_i_j and q_i must be of the same size.")
        self.M_i_j, self.q_i = M_i_j, q_i
        self.nbi = len(q_i)
        if z_names_i is None :
            z_names_i = ['z_'+str(i+1) for i in range(self.nbi)]
        if w_names_i is None :
            w_names_i = ['w_'+str(i+1) for i in range(self.nbi)]
        self.z_names_i, self.w_names_i = z_names_i, w_names_i

    def qp_solve(self, silent=True, verbose=0):
        qp = grb.Model()
        if silent:
            qp.Params.OutputFlag = 0
        qp.Params.NonConvex = 2
        z = qp.addMVar(shape = self.nbi)
        w = qp.addMVar(shape = self.nbi)
        qp.addConstr(w - self.M_i_j @ z == self.q_i)
        qp.setObjective(z @ w, sense = grb.GRB.MINIMIZE)
        qp.optimize()
        zsol, wsol, obj = z.x, w.x, qp.ObjVal
        print('z.w =', obj)
        if verbose > 0:
            print('z =', zsol)
            print('w =', wsol)
        return zsol, wsol, obj

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

    def create_tableau(self, display=False):
        tab = Tableau(A_i_j = -np.block([self.M_i_j, np.ones((self.nbi,1))]),
                      d_i = self.q_i, c_j = None,
                      decision_var_names_j=self.z_names_i+['z_0'], slack_var_names_i=self.w_names_i)
        self.tableau = tab
        if display: tab.display()

    def initialize_basis(self, verbose=0, display=False):
        self.create_tableau()
        zis = self.tableau.decision_var_names_j
        wis = self.tableau.slack_var_names_i
        kent = 2*self.nbi # z_0 enters
        kdep = np.argmin(self.q_i) # w_istar departs
        self.tableau.update(kent, kdep)
        if verbose>0: print((wis+zis)[kent] + ' enters, ' + (wis+zis)[kdep] + ' departs')
        if display: self.tableau.display()
        return kdep

    def lemke_solve(self, verbose=0, maxit=100):
        counter = 0
        if all(self.q_i >= 0):
            print('==========')
            print('Solution found: trivial LCP (q >= 0).')
            zsol, wsol = np.zeros(self.nbi), self.q_i
            return zsol, wsol
        kdep = self.initialize_basis(verbose=verbose-1)
        zis = self.tableau.decision_var_names_j
        wis = self.tableau.slack_var_names_i
        complements = list(self.nbi+np.arange(self.nbi)) + list(np.arange(self.nbi))
        while counter < maxit:
            counter += 1
            kent = complements[kdep]
            kdep = self.tableau.determine_departing(kent)
            if kdep is None:
                break
            if verbose > 1:
                print('Basis: ', [(wis+zis)[i] for i in self.tableau.k_b])
                print((wis+zis)[kent], 'enters,', (wis+zis)[kdep], 'departs')
            self.tableau.update(kent, kdep)
            if kdep == 2*self.nbi:
                break
        print('==========')
        if kdep == 2*self.nbi:
            print('Solution found: z_0 departed basis.')
            zsol, _, _ = self.tableau.solution()
            wsol = self.M_i_j @ zsol[:-1] + self.q_i
            if verbose > 0:
                print('Complementarity: z.w = ' + str(zsol[:-1] @ wsol))
                for i in range(self.nbi): print('z_'+str(i+1)+' = ' + str(zsol[i]))
                for i in range(self.nbi): print('w_'+str(i+1)+' = ' + str(wsol[i]))
            return zsol[:-1], wsol
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

    def create_penalty_game(zero_sum=False):
        penalty_data = np.array([[53.21, 71.35, 93.80],
                                 [90.26, 42.81, 86.12],
                                 [96.88, 100.0, 75.43]])
        if zero_sum:
            zerosum_game = Bimatrix_game(A_i_j = penalty_data, B_i_j = 100 - penalty_data)
            return zerosum_game
        else:
            nonzerosum_game = Bimatrix_game(A_i_j = penalty_data,
                                            B_i_j = np.array([[150, 100, 100],
                                                              [100, 150, 100],
                                                              [100, 100, 150]]) - penalty_data)
            return nonzerosum_game

    def is_NashEq(self, p_i, q_j, tol=1e-5):
        for i in range(self.nbi):
            if np.eye(self.nbi)[i] @ self.A_i_j @ q_j > p_i @ self.A_i_j @ q_j + tol:
                print('Pure strategy i =', i, 'beats p_i.')
                return False
        for j in range(self.nbj):
            if p_i @ self.B_i_j @ np.eye(self.nbj)[j] > p_i @ self.B_i_j @ q_j + tol:
                print('Pure strategy j =', j, 'beats q_j.')
                return False
        return True

    def zero_sum_solve(self, verbose=0):
        return Matrix_game(self.A_i_j).solve(verbose)

    def mangasarian_stone_solve(self, verbose=0):
        model=grb.Model()
        model.Params.OutputFlag = 0
        model.params.NonConvex = 2
        p_i = model.addMVar(shape = self.nbi)
        q_j = model.addMVar(shape = self.nbj)
        α = model.addMVar(shape = 1, lb = -grb.GRB.INFINITY)
        β = model.addMVar(shape = 1, lb = -grb.GRB.INFINITY)
        model.setObjective(α + β - p_i @ (self.A_i_j + self.B_i_j) @ q_j, sense = grb.GRB.MINIMIZE)
        model.addConstr(α * np.ones((self.nbi,1)) - self.A_i_j @ q_j >= 0)
        model.addConstr(β * np.ones((self.nbj,1)) - self.B_i_j.T @ p_i >= 0)
        model.addConstr(p_i.sum() == 1)
        model.addConstr(q_j.sum() == 1)
        model.optimize()
        sol = np.array(model.getAttr('x'))
        if verbose > 0: print('p_i =', sol[:self.nbi], '\nq_j =', sol[self.nbi:(self.nbi+self.nbj)])
        return {'p_i': sol[:self.nbi], 'q_j': sol[self.nbi:(self.nbi+self.nbj)],
                'val1': sol[-2], 'val2': sol[-1]}

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
        return {'p_i': p_i, 'q_j': q_j,
                'val1': α + np.min(self.A_i_j) - 1,
                'val2': β + np.min(self.B_i_j) - 1}


class LTU_problem:
    def __init__(self, Φ_x_y, λ_x_y = None, n_x = None, m_y = None):
        self.Φ_x_y = Φ_x_y
        self.nbx, self.nby = Φ_x_y.shape
        if λ_x_y is None: λ_x_y = np.ones(self.nbx, self.nby)/2  # λ_x_y = 1/2 if not provided (TU)
        self.λ_x_y = λ_x_y
        if n_x is None: n_x = np.ones(self.nbx)
        if m_y is None: m_y = np.ones(self.nby)
        self.n_x, self.m_y = n_x, m_y

    def to_LCP(self):
        M_X = np.kron( np.eye(self.nbx), np.ones(self.nby).T )
        M_Y = np.kron( np.ones(self.nbx).T, np.eye(self.nby) )
        Λ = np.diag(self.λ_x_y.flatten())
        return LCP(M_i_j = np.block([[np.zeros((self.nbx*self.nby, self.nbx*self.nby)), Λ @ M_X.T, (np.eye(self.nbx*self.nby)-Λ) @ M_Y.T],
                                     [-M_X, np.zeros((self.nbx, self.nbx)), np.zeros((self.nbx, self.nby))],
                                     [-M_Y, np.zeros((self.nby, self.nbx)), np.zeros((self.nby, self.nby))]]),
                   q_i = np.concatenate([-self.Φ_x_y.flatten()/2, self.n_x, self.m_y]))

    def is_solution(self, μ, u, v, tol=10e-6, verbose=0):
        self_LCP = self.to_LCP()
        z = np.concatenate([μ.flatten(), u, v])
        w = self_LCP.M_i_j @ z + self_LCP.q_i
        if verbose > 0:
            for i in range(self_LCP.nbi): print( 'z_' + str(i+1) + ' = ' + str(z[i].round(int(-np.log10(tol)))) )
            for i in range(self_LCP.nbi): print( 'w_' + str(i+1) + ' = ' + str(w[i].round(int(-np.log10(tol)))) )
            print( 'z . w = ' + str((z @ w).round(int(-np.log10(tol)))) )
        is_sol = all(z >= -tol) and all(w >= -tol) and (abs(z @ w) <= tol)
        return is_sol

    def bimatrix_game(self):
        if any(self.Φ_x_y.flatten() <= 0):
            print('LTU problem with nonpositive outputs.')
            return
        a_xy_x = -self.λ_x_y / (self.n_x.reshape(-1, 1) * self.Φ_x_y)
        b_xy_x = 1/(2 * self.n_x.reshape(-1, 1) * self.Φ_x_y)
        a_xy_y = -(1-self.λ_x_y) / (self.m_y * self.Φ_x_y)
        b_xy_y = 1/(2 * self.m_y * self.Φ_x_y)
        M_X = np.kron( np.eye(self.nbx), np.ones(self.nby).T )
        M_Y = np.kron( np.ones(self.nbx).T, np.eye(self.nby) )
        A = np.block([np.diag(a_xy_x.flatten()) @ M_X.T, np.diag(a_xy_y.flatten()) @ M_Y.T])
        B = np.block([np.diag(b_xy_x.flatten()) @ M_X.T, np.diag(b_xy_y.flatten()) @ M_Y.T])
        return Bimatrix_game(A, B)

    def solve(self, method='lemke_howson', verbose=0):
        self_game = self.bimatrix_game()
        if method == 'lemke_howson':
            sol_game = self_game.lemke_howson_solve(verbose=verbose)
        elif method == 'mangasarian_stone':
            sol_game = self_game.mangasarian_stone_solve()
        else:
            print('Method should be either \'lemke_howson\' or \'mangasarian_stone\'')
            return
        p_x_y = sol_game['p_i'].reshape((self.nbx, self.nby))
        q_x = sol_game['q_j'][:self.nbx]
        q_y = sol_game['q_j'][self.nbx:]
        μsol = p_x_y / ( 2 * self.Φ_x_y * (sol_game['p_i'] @ self_game.B_i_j @ sol_game['q_j']) )
        usol = q_x / ( 2 * self.n_x * (sol_game['p_i'] @ (-self_game.A_i_j) @ sol_game['q_j']) )
        vsol = q_y / ( 2 * self.m_y * (sol_game['p_i'] @ (-self_game.A_i_j) @ sol_game['q_j']) )
        if verbose>0:
            print('Matching matrix:'), print(μsol.round(2))
            print('\nUtilities:')
            for x in range(self.nbx): print('u_' + str(x+1) + ' = ' + str(usol[x].round(2)))
            for y in range(self.nby): print('v_' + str(y+1) + ' = ' + str(vsol[y].round(2)))
        return μsol, usol, vsol



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
