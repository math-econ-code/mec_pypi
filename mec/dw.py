import numpy as np
import gurobipy as grb
import time
import scipy.sparse as sp


def _as_integer_counts(name, values, tol=1e-6):
    values = np.asarray(values)
    rounded = np.rint(values)
    if np.any(values < -tol):
        raise ValueError(f"{name} contains negative counts.")
    if not np.all(np.abs(values - rounded) <= tol):
        raise ValueError(f"{name} must contain integer counts.")
    counts = rounded.astype(int)
    if np.any(counts < 0):
        raise ValueError(f"{name} contains negative counts.")
    return counts



class TUMatching:
    
    def __init__(self, Phi_x_y, eps_i_y, eta_x_j, eps_i_0, eta_0_j, delta_i_x, delta_j_y):
        self.I, self.Y = eps_i_y.shape
        self.X, self.J = eta_x_j.shape
        
        self.Phi_x_y = Phi_x_y
        self.eps_i_y, self.eps_i_0 = eps_i_y, eps_i_0
        self.eta_x_j, self.eta_0_j = eta_x_j, eta_0_j
        
        self.x_i, self.y_j = delta_i_x.argmax(axis=1), delta_j_y.argmax(axis=1)
        self.delta_i_x, self.delta_j_y = sp.csr_matrix(delta_i_x), sp.csr_matrix(delta_j_y)
        
        self.alpha_i_y = Phi_x_y[self.x_i,:] / 2 + eps_i_y
        self.gamma_x_j = Phi_x_y[:,self.y_j] / 2 + eta_x_j
        self.first_j_by_y = np.full(self.Y, -1, dtype=int)
        for j, y in enumerate(self.y_j):
            if self.first_j_by_y[y] == -1:
                self.first_j_by_y[y] = j
        
        self.Y_i_y = np.zeros((self.I, self.Y), dtype=bool)
        self.X_j_x = np.zeros((self.J, self.X), dtype=bool)
        
        
    def gurobi_solve(self, verbose=0):
        '''Solve the full type-aggregated equilibrium where every agent can choose every partner type.'''
        t_start = time.perf_counter()

        m = grb.Model("UnrestrictedMP")
        if verbose==0: m.Params.OutputFlag = 0

        pi_i_y = m.addMVar((self.I, self.Y), lb=0.0)
        pi_x_j = m.addMVar((self.X, self.J), lb=0.0)
        pi_i_0 = m.addMVar(self.I, lb=0.0)
        pi_0_j = m.addMVar(self.J, lb=0.0)

        obj = pi_i_0.T @ self.eps_i_0 + pi_0_j.T @ self.eta_0_j + (pi_i_y * self.alpha_i_y).sum() + (pi_x_j * self.gamma_x_j).sum()
        m.setObjective(obj, grb.GRB.MAXIMIZE)

        m.addConstr( pi_i_0 + pi_i_y.sum(axis=1) == 1.0 )
        m.addConstr( pi_0_j + pi_x_j.T.sum(axis=1) == 1.0 )
        m.addConstr( self.delta_i_x.T @ pi_i_y == pi_x_j @ self.delta_j_y )

        build_time = time.perf_counter() - t_start

        m.optimize()

        pi = np.array(m.x[:self.I*self.Y+self.X*self.J])
        pi_i_y = pi[:self.I*self.Y].reshape((self.I, self.Y))
        pi_x_j = pi[self.I*self.Y:].reshape((self.X, self.J))

        t = time.perf_counter() - t_start
        if verbose > 0:
            print("\nTotal time:", t)

        return pi_i_y, pi_x_j, t, build_time, m

        
    def basic_feasible_solution(self, verbose=0):
        '''Seed the restricted market with one representative man per type who can meet any woman type.'''
        for y in range(self.Y):
            j = self.first_j_by_y[y]
            if j == -1:
                raise ValueError(f"No man of type y = {y}")
            self.X_j_x[j, :] = True
            if verbose > 0:
                print(f"Type y = {y}: designated j_{y} = {j}")
                
                
    def build_rmp(self, verbose=0):
        '''Build the restricted market where agents may only match with types already in their choice sets.'''
        m = grb.Model("RestrictedMP")
        m.Params.OutputFlag  = verbose > 0
        m.Params.UpdateMode  = 1
        
        m.setObjective(0, grb.GRB.MAXIMIZE)

        row_i, row_j, linking_x_y = {}, {}, {}
        for x in range(self.X):
            for y in range(self.Y):
                linking_x_y[x, y] = m.addConstr(grb.LinExpr() == 0.0, name=f"linking_{x}_{y}")

        pi_i_0, pi_i_y = {}, {}
        for i in range(self.I):
            pi_i_0[i] = m.addVar(obj=self.eps_i_0[i], lb=0.0)
            row_i[i] = m.addConstr(pi_i_0[i] == 1.0, name=f"row_i_{i}")
        active_i, active_y = np.nonzero(self.Y_i_y)
        for i, y in zip(active_i, active_y):
            x = self.x_i[i].item()
            col = grb.Column()
            col.addTerms(1.0, row_i[i])
            col.addTerms(1.0, linking_x_y[x, y])
            pi_i_y[i,y] = m.addVar(obj=self.alpha_i_y[i,y], lb=0.0, column=col, name=f"L_{i}_{y}")

        pi_0_j, pi_x_j = {}, {}
        for j in range(self.J):
            pi_0_j[j] = m.addVar(obj=self.eta_0_j[j], lb=0.0)
            row_j[j] = m.addConstr(pi_0_j[j] == 1.0, name=f"row_j_{j}")
        active_j, active_x = np.nonzero(self.X_j_x)
        for j, x in zip(active_j, active_x):
            y = self.y_j[j].item()
            col = grb.Column()
            col.addTerms(1.0, row_j[j])
            col.addTerms(-1.0, linking_x_y[x, y])
            pi_x_j[x,j] = m.addVar(obj=self.gamma_x_j[x,j], lb=0.0, column=col, name=f"R_{x}_{j}")

        self.pi_i_0, self.pi_i_y = pi_i_0, pi_i_y
        self.pi_0_j, self.pi_x_j = pi_0_j, pi_x_j
        self.row_i, self.row_j, self.linking_x_y = row_i, row_j, linking_x_y
        self.linking_keys_xy = list(linking_x_y.keys())
        self.linking_constrs_xy = [linking_x_y[xy] for xy in self.linking_keys_xy]
        self.linking_x_idx, self.linking_y_idx = map(np.array, zip(*self.linking_keys_xy))
        self.m = m

        return m

  
    def find_improved_reduced_cost(self, rc_tol=1e-6, verbose=0, topk=1):
        '''Ask each agent which excluded type would most improve their supported utility.'''
    
        u_i = np.array(self.m.getAttr("Pi", list(self.row_i.values())))  # (I,)
        v_j = np.array(self.m.getAttr("Pi", list(self.row_j.values())))  # (J,)

        t_x_y = np.zeros((self.X, self.Y), dtype=np.float32)
        t_xy = np.array(self.m.getAttr("Pi", self.linking_constrs_xy))
        t_x_y[self.linking_x_idx, self.linking_y_idx] = t_xy

        rc_i = self.alpha_i_y.astype(np.float32) - t_x_y[self.x_i, :] - u_i[:, None]
        rc_i[self.Y_i_y] = -np.inf  # Block columns already in RMP

        rc_j = self.gamma_x_j.astype(np.float32) + t_x_y[:, self.y_j] - v_j[None, :]
        rc_j[self.X_j_x.T] = -np.inf

        new_columns_i_y, new_columns_j_x = [], []

        # Pick best (or top-k) 
        if topk == 1:
            best_y = rc_i.argmax(axis=1)
            best_rc_i = rc_i[np.arange(self.I), best_y]
            add_i = np.where(best_rc_i > rc_tol)[0]
            for i in add_i:
                y = int(best_y[i])
                self.Y_i_y[i, y] = True
                new_columns_i_y.append((i, y))
                if verbose > 1:
                    print(f"i={i}: add y={y}, rc={best_rc_i[i]:.3g}")
        else:
            # Take all y with rc > tol, but cap to topk per i
            idx_i, idx_y = np.where(rc_i > rc_tol)
            if idx_i.size:
                order = np.argsort(-rc_i[idx_i, idx_y])
                idx_i, idx_y = idx_i[order], idx_y[order]
                seen = np.zeros(self.I, dtype=np.int32)
                for i, y in zip(idx_i, idx_y):
                    if seen[i] >= topk:
                        continue
                    self.Y_i_y[int(i), int(y)] = True
                    new_columns_i_y.append((int(i), int(y)))
                    seen[i] += 1

        if topk == 1:
            best_x = rc_j.argmax(axis=0)
            best_rc_j = rc_j[best_x, np.arange(self.J)]
            add_j = np.where(best_rc_j > rc_tol)[0]
            for j in add_j:
                x = int(best_x[j])
                self.X_j_x[j, x] = True
                new_columns_j_x.append((j, x))
                if verbose > 1:
                    print(f"j={j}: add x={x}, rc={best_rc_j[j]:.3g}")
        else:
            idx_x, idx_j = np.where(rc_j > rc_tol)
            if idx_x.size:
                order = np.argsort(-rc_j[idx_x, idx_j])
                idx_x, idx_j = idx_x[order], idx_j[order]
                seen = np.zeros(self.J, dtype=np.int32)
                for x, j in zip(idx_x, idx_j):
                    if seen[j] >= topk:
                        continue
                    self.X_j_x[int(j), int(x)] = True
                    new_columns_j_x.append((int(j), int(x)))
                    seen[j] += 1

        return new_columns_i_y, new_columns_j_x


    def update_rmp(self, new_columns_i_y, new_columns_j_x):
        '''Add the newly desired type options as feasible matches in the restricted market.'''
        for (i, y) in new_columns_i_y:
            x = self.x_i[i].item()
            col = grb.Column()
            col.addTerms(1.0, self.row_i[i])
            col.addTerms(1.0, self.linking_x_y[x,y])
            self.pi_i_y[i,y] = self.m.addVar(obj=self.alpha_i_y[i,y], lb=0.0, column=col, name=f"L_{i}_{y}")
        for (j, x) in new_columns_j_x:
            y = self.y_j[j].item()
            col = grb.Column()
            col.addTerms(1.0, self.row_j[j])
            col.addTerms(-1.0, self.linking_x_y[x,y])
            self.pi_x_j[x,j] = self.m.addVar(obj=self.gamma_x_j[x,j], lb=0.0, column=col, name=f"R_{x}_{j}")
        return
    

    def rroa_solve(self, max_iter=100, rc_tol=1e-6, verbose=0):
        '''Repeatedly solve the restricted market until transfers make all missing choices unattractive.'''
        start_time = time.perf_counter()  # Add timing
        total_lp_iterations = 0           # Track LP iterations

        self.basic_feasible_solution()
        self.build_rmp()
        build_time = time.perf_counter() - start_time 

        self.m.Params.OutputFlag = 0
        self.m.Params.Threads = 0
        self.m.optimize()

        total_lp_iterations += self.m.IterCount  # Count initial solve
        history = [self.m.ObjVal]
        if verbose > 0:
            print(f"Iter 0: obj = {self.m.ObjVal:.6f}  (initial BFS)")

        self.m.setParam("Presolve", 0)
        self.m.Params.LPWarmStart = 2

        for k in range(max_iter):
            new_columns_i_y, new_columns_j_x = self.find_improved_reduced_cost(rc_tol=rc_tol)

            if not new_columns_i_y and not new_columns_j_x:     # stop condition
                if verbose > 0:
                    print(f"Iter {k+1:2d}: no positive reduced cost – optimal.")
                break

            self.update_rmp(new_columns_i_y, new_columns_j_x)
            
            self.m.optimize()
            if verbose:
                print("====")
                print(self.m.ObjVal)
            total_lp_iterations += self.m.IterCount  # Count iterations
            history.append(self.m.ObjVal)

        total_time = time.perf_counter() - start_time
        total_vars = int(self.I + self.J + self.X_j_x.sum() + self.Y_i_y.sum())
ß
        return history, total_time, total_vars, total_lp_iterations, build_time, self.m



class TUMatchingEstimation:

    def __init__(self, mu_x_y, mu_x0, mu_0y, phi_x_y_k, eps_i_0=None, eta_0_j=None, eps_i_y=None, eta_x_j=None, seed=0):
        mu_x_y = _as_integer_counts("mu_x_y", mu_x_y)
        mu_x0 = _as_integer_counts("mu_x0", mu_x0)
        mu_0y = _as_integer_counts("mu_0y", mu_0y)
        phi_x_y_k = np.asarray(phi_x_y_k)
        self.X, self.Y, self.K = phi_x_y_k.shape
        if mu_x_y.shape != (self.X, self.Y):
            raise ValueError("mu_x_y must have shape (X, Y).")
        if mu_x0.shape != (self.X,):
            raise ValueError("mu_x0 must have shape (X,).")
        if mu_0y.shape != (self.Y,):
            raise ValueError("mu_0y must have shape (Y,).")

        self.mu_x_y, self.mu_x0, self.mu_0y = mu_x_y, mu_x0, mu_0y
        self.phi_x_y_k = phi_x_y_k

        # construct an individual matching pi_i_y and pi_x_j consistent with aggregate mu_x_y, mu_x0, mu_0y
        n_x = mu_x_y.sum(axis=1) + mu_x0
        m_y = mu_x_y.sum(axis=0) + mu_0y
        self.I, self.J = int(n_x.sum()), int(m_y.sum())
        
        x_i = np.repeat(np.arange(self.X), n_x) # repeat 0 times n_x[0], 1 times n_x[1], etc.
        y_j = np.repeat(np.arange(self.Y), m_y)

        self.x_i = x_i
        self.y_j = y_j
        self.delta_i_x = np.eye(self.X)[x_i]
        self.delta_j_y = np.eye(self.Y)[y_j]
        
        self.pi_i_y = np.zeros((self.I, self.Y))
        self.pi_x_j = np.zeros((self.X, self.J))
        
        i = 0
        for x in range(self.X):
            for y in range(self.Y):
                self.pi_i_y[i:(i+mu_x_y[x,y]),y] = 1
                i += mu_x_y[x,y]
            i += mu_x0[x]
        
        j = 0
        for y in range(self.Y):
            for x in range(self.X):
                self.pi_x_j[x,j:(j+mu_x_y[x,y])] = 1
                j += mu_x_y[x,y]
            j += mu_0y[y]

        self.eps_i_0 = eps_i_0
        self.eta_0_j = eta_0_j
        self.eps_i_y = eps_i_y
        self.eta_x_j = eta_x_j
        np.random.seed(seed)
        if eps_i_0 is None:
            self.eps_i_0 = np.random.normal(0, 1, size=(self.I)) / 10
        if eta_0_j is None:
            self.eta_0_j = np.random.normal(0, 1, size=(self.J)) / 10
        if eps_i_y is None:
            self.eps_i_y = np.random.normal(0, 1, size=(self.I,self.Y)) / 10
        if eta_x_j is None:
            self.eta_x_j = np.random.normal(0, 1, size=(self.X,self.J)) / 10

        self.Y_i_y = np.zeros((self.I, self.Y), dtype=bool)
        self.X_j_x = np.zeros((self.J, self.X), dtype=bool)
        self.active_i_0 = np.zeros(self.I, dtype=bool)
        self.active_0_j = np.zeros(self.J, dtype=bool)


    def gurobi_solve(self, verbose=0):
        '''Solve the simulated inverse market with moment prices chosen to support the observed matching.'''
        t_start = time.perf_counter()

        m = grb.Model("UnrestrictedMP")
        if verbose == 0: m.Params.OutputFlag = 0

        pi_i_y = m.addMVar((self.I, self.Y), lb=0.0)
        pi_x_j = m.addMVar((self.X, self.J), lb=0.0)
        pi_i_0 = m.addMVar(self.I, lb=0.0)
        pi_0_j = m.addMVar(self.J, lb=0.0)

        obj = pi_i_0.T @ self.eps_i_0 + pi_0_j.T @ self.eta_0_j + (pi_i_y * self.eps_i_y).sum() + (pi_x_j * self.eta_x_j).sum()

        m.setObjective(obj, grb.GRB.MAXIMIZE)

        m.addConstr(pi_i_0 + pi_i_y.sum(axis=1) == 1.0)
        m.addConstr(pi_0_j + pi_x_j.T.sum(axis=1) == 1.0)
        m.addConstr(self.delta_i_x.T @ pi_i_y == pi_x_j @ self.delta_j_y)

        mu_lambda_x_y = ( self.delta_i_x.T @ pi_i_y + pi_x_j @ self.delta_j_y ) / 2
        m.addConstr( (self.mu_x_y[:,:,None] * self.phi_x_y_k).sum(axis=(0,1)) == (mu_lambda_x_y[:,:,None] * self.phi_x_y_k).sum(axis=(0,1)) )

        build_time = time.perf_counter() - t_start
        m.optimize()

        self.m = m
        lambda_k = np.array(m.Pi[-self.K:])
        t = time.perf_counter() - t_start
        if verbose > 0:
            print("\nTotal time:", t)

        return lambda_k, t, build_time, m

  
    def basic_feasible_solution(self):
        '''Start the simulated inverse market from the observed aggregate matches and singles.'''
        idx_i_by_x = [list(np.where(self.x_i == x)[0]) for x in range(self.X)]
        idx_j_by_y = [list(np.where(self.y_j == y)[0]) for y in range(self.Y)]

        self.Y_i_y[:, :] = False
        self.X_j_x[:, :] = False
        self.active_i_0[:] = False
        self.active_0_j[:] = False

        for x in range(self.X):
            for y in range(self.Y):
                c = int(self.mu_x_y[x, y])
                for _ in range(c):
                    i = idx_i_by_x[x].pop()
                    j = idx_j_by_y[y].pop()
                    self.Y_i_y[i, y] = True
                    self.X_j_x[j, x] = True
                    
        for x in range(self.X):
            for i in idx_i_by_x[x]:
                self.active_i_0[i] = True
        for y in range(self.Y):
            for j in idx_j_by_y[y]:
                self.active_0_j[j] = True
                
        num_extra_singles = self.K + self.X*self.Y
        i = 0
        while num_extra_singles > 0 and i < self.I:
            if not self.active_i_0[i]:
                self.active_i_0[i] = True
                num_extra_singles -= 1
            i += 1
        j = 0
        while num_extra_singles > 0 and j < self.J:
            if not self.active_0_j[j]:
                self.active_0_j[j] = True
                num_extra_singles -= 1
            j += 1
        if num_extra_singles > 0:
            raise ValueError("Not enough inactive singlehood columns to complete the initial basis.")
        return

  
    def build_rmp(self, verbose=0):
        '''Build the restricted simulated market around current choices and moment constraints.'''
        m = grb.Model("RestrictedMP")
        m.Params.OutputFlag = verbose > 0
        m.Params.UpdateMode = 1

        m.setObjective(0, grb.GRB.MAXIMIZE)

        row_i, row_j, linking_x_y, linking_k = {}, {}, {}, {}
        for i in range(self.I):
            row_i[i] = m.addConstr(grb.LinExpr() == 1.0, name=f"row_i_{i}")
        for j in range(self.J):
            row_j[j] = m.addConstr(grb.LinExpr() == 1.0, name=f"row_j_{j}")
        for x in range(self.X):
            for y in range(self.Y):
                linking_x_y[x, y] = m.addConstr(grb.LinExpr() == 0.0, name=f"linking_{x}_{y}")

        lambda_i_0, lambda_i_y = {}, {}
        lambda_0_j, lambda_x_j = {}, {}

        rhs_k = (self.phi_x_y_k * self.mu_x_y[:, :, None]).sum(axis=(0, 1))
        self.linking_k_list = []
        for k in range(self.K):
            constr = m.addConstr(grb.LinExpr() == rhs_k[k], name=f"linking_{k}")
            linking_k[k] = constr
            self.linking_k_list.append(constr)

        for i in np.flatnonzero(self.active_i_0):
            col = grb.Column()
            col.addTerms(1.0, row_i[i])
            lambda_i_0[i] = m.addVar(obj=self.eps_i_0[i], lb=0.0, column=col, name=f"L_{i}_0")

        active_i, active_y = np.nonzero(self.Y_i_y)
        for i, y in zip(active_i, active_y):
            x = int(self.x_i[i])
            col = grb.Column()
            col.addTerms(1.0, row_i[i])
            col.addTerms(1.0, linking_x_y[x, y])
            for k in range(self.K):
                col.addTerms(self.phi_x_y_k[x, y, k] / 2, linking_k[k])
            lambda_i_y[i, y] = m.addVar(obj=self.eps_i_y[i, y], lb=0.0, column=col, name=f"L_{i}_{y}")

        for j in np.flatnonzero(self.active_0_j):
            col = grb.Column()
            col.addTerms(1.0, row_j[j])
            lambda_0_j[j] = m.addVar(obj=self.eta_0_j[j], lb=0.0, column=col, name=f"R_0_{j}")

        active_j, active_x = np.nonzero(self.X_j_x)
        for j, x in zip(active_j, active_x):
            y = int(self.y_j[j])
            col = grb.Column()
            col.addTerms(1.0, row_j[j])
            col.addTerms(-1.0, linking_x_y[x, y])
            for k in range(self.K):
                col.addTerms(self.phi_x_y_k[x, y, k] / 2, linking_k[k])
            lambda_x_j[x, j] = m.addVar(obj=self.eta_x_j[x, j], lb=0.0, column=col, name=f"R_{x}_{j}")

        self.lambda_i_0, self.lambda_i_y = lambda_i_0, lambda_i_y
        self.lambda_0_j, self.lambda_x_j = lambda_0_j, lambda_x_j
        self.row_i, self.row_j, self.linking_x_y, self.linking_k = row_i, row_j, linking_x_y, linking_k
        self.row_i_list = [row_i[i] for i in range(self.I)]
        self.row_j_list = [row_j[j] for j in range(self.J)]
        self.linking_keys_xy = list(linking_x_y.keys())
        self.linking_constrs_xy = [linking_x_y[xy] for xy in self.linking_keys_xy]
        self.linking_x_idx, self.linking_y_idx = map(np.array, zip(*self.linking_keys_xy))
        self.linking_k_list = [linking_k[k] for k in range(self.K)]
        self.m = m

        return m

  
    def get_lambdas(self):
        return -np.array([constr.Pi for constr in self.linking_k_list])

  
    def find_improved_reduced_cost(self, rc_tol=1e-6, verbose=0):
        '''Ask simulated agents whether missing choices improve utility at current transfers and moment prices.'''
        u_i = np.array(self.m.getAttr("Pi", self.row_i_list))
        v_j = np.array(self.m.getAttr("Pi", self.row_j_list))
        t_x_y = np.zeros((self.X, self.Y)) # matrix access is faster than dictionary for this size
        t_xy = np.array(self.m.getAttr("Pi", self.linking_constrs_xy))
        t_x_y[self.linking_x_idx, self.linking_y_idx] = t_xy
        lambda_k = -np.array(self.m.getAttr("Pi", self.linking_k_list))

        Phi_lambda_x_y = self.phi_x_y_k @ lambda_k

        u_i_y = Phi_lambda_x_y[self.x_i, :] / 2 - t_x_y[self.x_i, :] + self.eps_i_y
        rc_i_y = u_i_y - u_i[:, None]
        rc_i_y[self.Y_i_y] = -np.inf
        y_star_i = rc_i_y.argmax(axis=1)
        best_rc_i = rc_i_y[np.arange(self.I), y_star_i]

        rc_i_0 = self.eps_i_0 - u_i
        use_i_0 = (~self.active_i_0) & (rc_i_0 > best_rc_i)

        add_i = np.flatnonzero((~use_i_0) & (best_rc_i > rc_tol))
        add_y = y_star_i[add_i].astype(int)
        self.Y_i_y[add_i, add_y] = True
        new_columns_i_y = list(zip(add_i.astype(int).tolist(), add_y.tolist()))
        add_i_0 = np.flatnonzero(use_i_0 & (rc_i_0 > rc_tol)).astype(int)
        self.active_i_0[add_i_0] = True
        new_columns_i_0 = add_i_0.tolist()
        if verbose > 1:
            for i, y in new_columns_i_y:
                print(f"Type y={y} entered choice set of i={i}. Reduced cost: {best_rc_i[i]:.2f}.")
            for i in new_columns_i_0:
                print(f"Singlehood entered choice set of i={i}. Reduced cost: {rc_i_0[i]:.2f}.")

        v_x_j = Phi_lambda_x_y[:, self.y_j] / 2 + t_x_y[:, self.y_j] + self.eta_x_j
        rc_x_j = v_x_j - v_j[None, :]
        rc_x_j[self.X_j_x.T] = -np.inf
        x_star_j = rc_x_j.argmax(axis=0)
        best_rc_j = rc_x_j[x_star_j, np.arange(self.J)]

        rc_0_j = self.eta_0_j - v_j
        use_0_j = (~self.active_0_j) & (rc_0_j > best_rc_j)

        add_j = np.flatnonzero((~use_0_j) & (best_rc_j > rc_tol))
        add_x = x_star_j[add_j].astype(int)
        self.X_j_x[add_j, add_x] = True
        new_columns_j_x = list(zip(add_j.astype(int).tolist(), add_x.tolist()))
        add_0_j = np.flatnonzero(use_0_j & (rc_0_j > rc_tol)).astype(int)
        self.active_0_j[add_0_j] = True
        new_columns_0_j = add_0_j.tolist()
        if verbose > 1:
            for j, x in new_columns_j_x:
                print(f"Type x={x} entered choice set of j={j}. Reduced cost: {best_rc_j[j]:.2f}.")
            for j in new_columns_0_j:
                print(f"Singlehood entered choice set of j={j}. Reduced cost: {rc_0_j[j]:.2f}.")

        return new_columns_i_y, new_columns_i_0, new_columns_j_x, new_columns_0_j

  
    def update_rmp(self, new_columns_i_y, new_columns_i_0, new_columns_j_x, new_columns_0_j):
        '''Add simulated choices that would improve the restricted inverse market.'''
        for i in new_columns_i_0:
            col = grb.Column()
            col.addTerms(1.0, self.row_i[i])
            self.lambda_i_0[i] = self.m.addVar(obj=self.eps_i_0[i], lb=0.0, column=col, name=f"L_{i}_0")
        for (i, y) in new_columns_i_y:
            x = int(self.x_i[i])
            col = grb.Column()
            col.addTerms(1.0, self.row_i[i])
            col.addTerms(1.0, self.linking_x_y[x, y])
            for k in range(self.K):
                col.addTerms(self.phi_x_y_k[x, y, k] / 2, self.linking_k[k])
            self.lambda_i_y[i, y] = self.m.addVar(obj=self.eps_i_y[i, y], lb=0.0, column=col, name=f"L_{i}_{y}")
        for j in new_columns_0_j:
            col = grb.Column()
            col.addTerms(1.0, self.row_j[j])
            self.lambda_0_j[j] = self.m.addVar(obj=self.eta_0_j[j], lb=0.0, column=col, name=f"R_0_{j}")
        for (j, x) in new_columns_j_x:
            y = int(self.y_j[j])
            col = grb.Column()
            col.addTerms(1.0, self.row_j[j])
            col.addTerms(-1.0, self.linking_x_y[x, y])
            for k in range(self.K):
                col.addTerms(self.phi_x_y_k[x, y, k] / 2, self.linking_k[k])
            self.lambda_x_j[x, j] = self.m.addVar(obj=self.eta_x_j[x, j], lb=0.0, column=col, name=f"R_{x}_{j}")
        return

  
    def rroa_solve(self, max_iter=100, rc_tol=1e-6, verbose=0):
        '''Expand the simulated inverse market until no missing choice can improve it.'''
        start_time = time.perf_counter()  # Add timing
        total_lp_iterations = 0  # Track LP iterations

        self.basic_feasible_solution()
        self.build_rmp()
        build_time = time.perf_counter() - start_time

        self.m.Params.OutputFlag = 0
        self.m.optimize()

        ###########
        if self.m.Status != grb.GRB.OPTIMAL:
            print(f"Model status: {self.m.Status}")
            if self.m.Status == grb.GRB.INFEASIBLE:
                print("Model is INFEASIBLE")
                self.m.computeIIS()
                self.m.write("model.ilp")
                for c in self.m.getConstrs():
                    if c.IISConstr:
                        print(f"Infeasible constraint: {c.ConstrName}  (sense {c.Sense}, rhs {c.RHS})")
                for v in self.m.getVars():
                    if v.IISLB: print(f"Infeasible LB: {v.VarName} >= {v.LB}")
                    if v.IISUB: print(f"Infeasible UB: {v.VarName} <= {v.UB}")
            elif self.m.Status == grb.GRB.UNBOUNDED:
                print("Model is UNBOUNDED")
            return [], 0, 0, 0, 0
        ###########

        total_lp_iterations += self.m.IterCount  # Count initial solve
        history = [self.m.ObjVal]
        if verbose > 0:
            print(f"Iter 0: obj = {self.m.ObjVal:.6f}  (initial BFS)")

        self.m.setParam("Presolve", 0)
        self.m.Params.LPWarmStart = 2

        for iter in range(max_iter):
            new_columns_i_y, new_columns_i_0, new_columns_j_x, new_columns_0_j = self.find_improved_reduced_cost(rc_tol=rc_tol)

            if not new_columns_i_y and not new_columns_i_0 and not new_columns_j_x and not new_columns_0_j:  # stop condition
                if verbose > 0:
                    print(f"Iter {iter + 1:2d}: no positive reduced cost – optimal.")
                break

            self.update_rmp(new_columns_i_y, new_columns_i_0, new_columns_j_x, new_columns_0_j)

            self.m.optimize()
            if verbose > 0:
                print("====")
                print(self.m.ObjVal)
            total_lp_iterations += self.m.IterCount  # Count iterations
            history.append(self.m.ObjVal)

        total_time = time.perf_counter() - start_time
        total_vars = int(self.active_i_0.sum() + self.active_0_j.sum() + self.Y_i_y.sum() + self.X_j_x.sum())

        return history, total_time, total_vars, total_lp_iterations, build_time


# Visualization tools

def visualize_results(unr_list, rroa_list):
    '''Compare the unrestricted and restricted markets to see whether they support the same surplus.'''
    _, _, t_unr, build_unr, m_unr = unr_list
    
    history_dict, rroa_time, rroa_vars, rroa_iterations, build_rroa, rroa_model = rroa_list
    rroa_obj = history_dict[-1] if history_dict else 0
    
    if m_unr.Status == grb.GRB.OPTIMAL:
        obj_unr = m_unr.ObjVal
        obj_diff = abs(obj_unr - rroa_obj)
        print(f"Objective values: BF={obj_unr:.6f}, RROA={rroa_obj:.6f}")
        print(f"\nMetrics comparison (from feedback):")

        print(f"{'Metric':<25} {'Full-column LP':<20} {'RROA':<20} {'Ratio/Speedup':<15}")
        print("-" * 80)
        print(f"{'Build time (s)':<25} {build_unr:<20.3f} {build_rroa:<20.3f} {build_unr/build_rroa:<15.2f}")
        print(f"{'Solve time (s)':<25} {t_unr:<20.3f} {rroa_time:<20.3f} {t_unr/rroa_time:<15.2f}")

        print(f"{'Iter (LP iterations)':<25} {int(m_unr.IterCount):<20d} {int(rroa_iterations):<20d} {int(m_unr.IterCount)/int(rroa_iterations):<15}")
        print(f"{'N_col (variables)':<25} {m_unr.NumVars:<20d} {rroa_vars:<20d} {m_unr.NumVars/rroa_vars:<15.1f}")
        if obj_diff < 1e-6:
            print(f"{'Obj (optimal value)':<25} {obj_unr:<20.6f} {rroa_obj:<20.6f} {'Match':<15}")
        else:
            print(f"{'Obj (optimal value)':<25} {obj_unr:<20.6f} {rroa_obj:<20.6f} {'Mismatch':<15}")
    else:
        print(f"Failed to solve! Status: {m_unr.Status}")
        obj_unr = None

    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    if history_dict:
        import pandas as pd
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        s = pd.Series(history_dict)
        s.plot(title="RROA Convergence", xlabel="Iteration", ylabel="Objective Value", marker='o', linewidth=2)
        plt.axhline(y=obj_unr, color='r', linestyle='--', label='Optimal (Full-column LP)', alpha=0.7)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
