import numpy as np
import gurobipy as grb
import time
import scipy.sparse as sp


class Dantzig_Wolfe_vector_approach():
    
    def __init__(self, Phi_x_y, eps_i_y, eta_x_j, eps_i_0, eta_0_j, delta_i_x, delta_j_y):
        
        self.I, self.Y = eps_i_y.shape
        self.X, self.J = eta_x_j.shape
        
        self.Phi_x_y = Phi_x_y
        self.eps_i_y, self.eps_i_0 = eps_i_y, eps_i_0
        self.eta_x_j, self.eta_0_j = eta_x_j, eta_0_j
        
        self.x_i, self.y_j = delta_i_x.argmax(axis=1), delta_j_y.argmax(axis=1)
        self.delta_i_x, self.delta_j_y = sp.csr_matrix(delta_i_x), sp.csr_matrix(delta_j_y)
        
        self.phi_i_y = Phi_x_y[self.x_i,:] / 2 + eps_i_y
        self.psi_x_j = Phi_x_y[:,self.y_j] / 2 + eta_x_j
    
        
        self.Y_i_y, self.X_j_x = np.zeros((self.I, self.Y)), np.zeros((self.J, self.X))
        if self.I* self.J < 1e9:
            self.Phi_i_j = delta_i_x @ Phi_x_y @ delta_j_y.T + eps_i_y @ delta_j_y.T + delta_i_x @ eta_x_j
    
    def naive_solve(self, verbose=0):
        t_start = time.perf_counter()

        m = grb.Model("NaiveLP")
        if verbose==0: m.Params.OutputFlag = 0

        pi_i_j = m.addMVar((I,J), lb=0.0)
        pi_i_0 = m.addMVar(I, lb=0.0)
        pi_0_j = m.addMVar(J, lb=0.0)

        m.setObjective( (pi_i_j * Phi_i_j).sum() + (pi_i_0 * self.eps_i_0).sum() + (pi_0_j * self.eta_0_j).sum(), grb.GRB.MAXIMIZE)
        m.addConstr( pi_i_0 + pi_i_j.sum(axis=1) == 1.0 )
        m.addConstr( pi_0_j + pi_i_j.T.sum(axis=1) == 1.0 )

        m.optimize()

        pi_i_j = np.array(m.x[:I*J]).reshape((I,J))

        t = time.perf_counter() - t_start
        print("\nTotal time:", t)

        return pi_i_j, t, m

    def unrestricted_solve(self, verbose=0):
        t_start = time.perf_counter()

        m = grb.Model("UnrestrictedMP")
        #m.Params.Method = 0
        if verbose==0: m.Params.OutputFlag = 0

        pi_i_y = m.addMVar((I,Y), lb=0.0)
        pi_x_j = m.addMVar((X,J), lb=0.0)
        pi_i_0 = m.addMVar(I, lb=0.0)
        pi_0_j = m.addMVar(J, lb=0.0)

        obj = pi_i_0.T @ self.eps_i_0 + pi_0_j.T @ self.eta_0_j + (pi_i_y * self.phi_i_y).sum() + (pi_x_j * self.psi_x_j).sum()
        m.setObjective(obj, grb.GRB.MAXIMIZE)

        m.addConstr( pi_i_0 + pi_i_y.sum(axis=1) == 1.0 )
        m.addConstr( pi_0_j + pi_x_j.T.sum(axis=1) == 1.0 )
        m.addConstr( self.delta_i_x.T @ pi_i_y == pi_x_j @ self.delta_j_y )

        build_time = time.perf_counter() - t_start

        m.optimize()

        pi = np.array(m.x[:self.I*self.Y+self.X*self.J])
        pi_i_y = pi[:self.I*self.Y].reshape((self.I,self.Y))
        pi_x_j = pi[self.I*self.Y:].reshape((self.X,self.J))

        t = time.perf_counter() - t_start
        # print("\nTotal time:", t)

        return pi_i_y, pi_x_j, t, build_time, m

    def basic_feasible_solution(self):
        for y in range(self.Y):
            j = 0
            while j < self.J and self.delta_j_y[j, y] == 0:
                j += 1
            if j == self.J:
                raise ValueError(f"No man of type y = {y}")
            self.X_j_x[j, :] = 1
            # print(f"Type y = {y}: designated j_{y} = {j}")

    def build_rmp(self, verbose=0):
        m = grb.Model("RMP")
        #m.Params.Method      = 0 # 0: primal simplex, 1: dual simplex, 2: barrier, ...
        m.Params.LPWarmStart = 2 
        m.Params.UpdateMode  = 1
        if verbose==0: m.Params.OutputFlag = 0

        # Objective
        lambda_i_0 = m.addMVar(self.I, lb=0.0)
        lambda_0_j = m.addMVar(self.J, lb=0.0)

        ub_i_y = np.where(self.Y_i_y, grb.GRB.INFINITY, 0.0)      
        ub_x_j = np.where(self.X_j_x.T, grb.GRB.INFINITY, 0.0)   
        lambda_i_y = m.addMVar((self.I, self.Y), lb=0.0, ub=ub_i_y)
        lambda_x_j = m.addMVar((self.X, self.J), lb=0.0, ub=ub_x_j)

        obj = lambda_i_0.T @ self.eps_i_0 + lambda_0_j.T @ self.eta_0_j + (lambda_i_y * self.phi_i_y).sum() + (lambda_x_j * self.psi_x_j).sum()
        m.setObjective(obj, grb.GRB.MAXIMIZE)

        # Row constraints
        m.addConstr( lambda_i_0 + lambda_i_y.sum(axis=1) == 1.0 )
        m.addConstr( lambda_0_j + lambda_x_j.T.sum(axis=1) == 1.0 )

        # Linking Constraints
        m.addConstr( self.delta_i_x.T @ lambda_i_y == lambda_x_j @ self.delta_j_y )

        return m, lambda_i_y, lambda_x_j
    

    def find_improved_reduced_cost(self, model, rc_tol=1e-6, verbose=0):
        dual_vals = model.getAttr("Pi", model.getConstrs())[:(self.I + self.J + self.X*self.Y)]
        u_i = np.array(dual_vals[:self.I])
        v_j = np.array(dual_vals[self.I:self.I+self.J])
        W_xy = np.array(dual_vals[self.I+self.J:self.I+self.J+self.X*self.Y]).reshape((self.X,self.Y))

        new_columns_i_y, new_columns_j_x = [], []

        for i in range(self.I):
            best_rc, best_y = -np.inf, None
            for y in range(self.Y):
                if y not in self.Y_i_y[i,:].nonzero()[0]:
                    rc = self.phi_i_y[i,y] - self.delta_i_x[i,:] @ W_xy[:,y] - u_i[i]
                    #if rc > rc_tol:
                    #    Y_i[i,y] = 1
                    #    new_columns_i_y.append((i,y))
                    if rc > best_rc:
                        best_rc, best_y = rc, y
                    if verbose>1: print(f"Agent i={i}, type y={y}. Reduced cost: {float(rc):.2f}.")

            if best_rc > rc_tol:
                self.Y_i_y[i,best_y] = 1
                new_columns_i_y.append((i,best_y))
                if verbose>0: print(f"Type y={best_y} entered choice set of i={i}. Reduced cost: {float(best_rc):.2f}.")

        for j in range(self.J):
            best_rc, best_x = -np.inf, None
            for x in range(self.X):
                if x not in self.X_j_x[j].nonzero()[0]:
                    rc = self.psi_x_j[x,j] + self.delta_j_y[j,:] @ W_xy[x,:].T - v_j[j]
                    #if rc > rc_tol:
                    #    X_j[j,x] = 1
                    #    new_columns_j_x.append((j,x))
                    if rc > best_rc:
                        best_rc, best_x = rc, x
                    if verbose>1: print(f"Agent j={j}, type x={x}. Reduced cost: {float(rc):.2f}.")

            if best_rc > rc_tol:
                self.X_j_x[j,best_x] = 1
                new_columns_j_x.append((j,best_x))
                if verbose>0: print(f"Type x={best_x} entered choice set of j={j}. Reduced cost: {float(best_rc):.2f}.")

        return new_columns_i_y, new_columns_j_x

    def update_rmp(self, model, new_columns_i_y, new_columns_j_x, lambda_i_y, lambda_x_j):
        for (i, y) in new_columns_i_y:
            lambda_i_y[i, y].UB = grb.GRB.INFINITY
        for (j, x) in new_columns_j_x:
            lambda_x_j[x, j].UB = grb.GRB.INFINITY
        return
    
    def column_generation(self, max_iter=100, rc_tol=1e-6, verbose=0):
        start_time = time.perf_counter()  # Add timing
        total_lp_iterations = 0           # Track LP iterations

        rmp, lambda_i_y, lambda_x_j = self.build_rmp()
        build_time = time.perf_counter() - start_time 

        rmp.Params.OutputFlag = 0
        rmp.optimize()

        total_lp_iterations += rmp.IterCount  # Count initial solve
        history = [rmp.ObjVal]
        print(f"Iter 0: obj = {rmp.ObjVal:.6f}  (initial BFS)")

        rmp.setParam("Presolve", 0)

        for k in range(max_iter):
            new_columns_i_y, new_columns_j_x = self.find_improved_reduced_cost(rmp, rc_tol=rc_tol)

            if not new_columns_i_y and not new_columns_j_x:     # stop condition
                print(f"Iter {k+1:2d}: no positive reduced cost – optimal.")
                break

            self.update_rmp(rmp, new_columns_i_y, new_columns_j_x, lambda_i_y, lambda_x_j)

            # primal_var = rmp.getVars()
            # last_sol = rmp.getAttr("X", primal_var)
            # rmp.setAttr("Start", primal_var, last_sol)
            rmp.optimize()
            if verbose:
                print("====")
                print(rmp.ObjVal)
            total_lp_iterations += rmp.IterCount  # Count iterations
            history.append(rmp.ObjVal)

        total_time = time.perf_counter() - start_time
        total_vars = int(self.I + self.J + self.X_j_x.sum() + self.Y_i_y.sum())

        return history, total_time, total_vars, total_lp_iterations, rmp, build_time


class Dantzig_Wolfe:
    
    def __init__(self, Phi_x_y, eps_i_y, eta_x_j, eps_i_0, eta_0_j, delta_i_x, delta_j_y):
        self.I, self.Y = eps_i_y.shape
        self.X, self.J = eta_x_j.shape
        
        self.Phi_x_y = Phi_x_y
        self.eps_i_y, self.eps_i_0 = eps_i_y, eps_i_0
        self.eta_x_j, self.eta_0_j = eta_x_j, eta_0_j
        
        self.x_i, self.y_j = delta_i_x.argmax(axis=1), delta_j_y.argmax(axis=1)
        self.delta_i_x, self.delta_j_y = sp.csr_matrix(delta_i_x), sp.csr_matrix(delta_j_y)
        
        self.phi_i_y = Phi_x_y[self.x_i,:] / 2 + eps_i_y
        self.psi_x_j = Phi_x_y[:,self.y_j] / 2 + eta_x_j
        
        self.Y_i_y = {i: set() for i in range(self.I)}
        self.X_j_x = {j: set() for j in range(self.J)}
        
        
    def unrestricted_solve(self, verbose=0):
        t_start = time.perf_counter()

        m = grb.Model("UnrestrictedMP")
        if verbose==0: m.Params.OutputFlag = 0

        pi_i_y = m.addMVar((self.I, self.Y), lb=0.0)
        pi_x_j = m.addMVar((self.X, self.J), lb=0.0)
        pi_i_0 = m.addMVar(self.I, lb=0.0)
        pi_0_j = m.addMVar(self.J, lb=0.0)

        obj = pi_i_0.T @ self.eps_i_0 + pi_0_j.T @ self.eta_0_j + (pi_i_y * self.phi_i_y).sum() + (pi_x_j * self.psi_x_j).sum()
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
        print("\nTotal time:", t)

        return pi_i_y, pi_x_j, t, build_time, m

        
    def basic_feasible_solution(self, verbose=0):
        for y in range(self.Y):
            j = 0
            while j < self.J and self.y_j[j] != y:
                j += 1
            if j == self.J:
                raise ValueError(f"No man of type y = {y}")
            self.X_j_x[j] = set(range(self.X))
            if (verbose > 0):
                print(f"Type y = {y}: designated j_{y} = {j}")
                
                
    def build_rmp(self, verbose=0):
        m = grb.Model("RMP")
        m.Params.OutputFlag  = verbose > 0
        m.Params.UpdateMode  = 1
        
        m.setObjective(0, grb.GRB.MAXIMIZE)

        row_i, row_j, linking_x_y = {}, {}, {}

        lambda_i_0, lambda_i_y = {}, {}
        for i in range(self.I):
            lambda_i_0[i] = m.addVar(obj=self.eps_i_0[i], lb=0.0)
            row_i[i] = m.addConstr(lambda_i_0[i] == 1.0, name=f"row_i_{i}")
            for y in self.Y_i_y[i]:
                col = grb.Column()
                col.addTerms(1.0, row_i[i])
                lambda_i_y[i,y] = m.addVar(obj=self.phi_i_y[i,y], lb=0.0, column=col, name=f"L_{i}_{y}")

        lambda_0_j, lambda_x_j = {}, {}
        for j in range(self.J):
            lambda_0_j[j] = m.addVar(obj=self.eta_0_j[j], lb=0.0)
            row_j[j] = m.addConstr(lambda_0_j[j] == 1.0, name=f"row_j_{j}")
            y = self.y_j[j].item()
            for x in self.X_j_x[j]:
                col = grb.Column()
                col.addTerms(1.0, row_j[j])
                lambda_x_j[x,j] = m.addVar(obj=self.psi_x_j[x,j], lb=0.0, column=col, name=f"R_{x}_{j}")
                linking_x_y[x,y] = m.addConstr(-lambda_x_j[x,j] == 0.0, name=f"linking_{x}_{y}")

        self.lambda_i_0, self.lambda_i_y = lambda_i_0, lambda_i_y
        self.lambda_0_j, self.lambda_x_j = lambda_0_j, lambda_x_j
        self.row_i, self.row_j, self.linking_x_y = row_i, row_j, linking_x_y
        self.m = m

        return m

  
    def find_improved_reduced_cost(self, rc_tol=1e-6, verbose=0, topk=1):
    
        u = np.array(self.m.getAttr("Pi", list(self.row_i.values())))  # (I,)
        v = np.array(self.m.getAttr("Pi", list(self.row_j.values())))  # (J,)

        # Build W[x,y] from linking constraints
        keys = list(self.linking_x_y.keys())  
        if keys:
            pis = np.array(self.m.getAttr("Pi", [self.linking_x_y[k] for k in keys]))
            xs, ys = zip(*keys)
            W = np.zeros((self.X, self.Y), dtype=np.float32)
            W[np.array(xs), np.array(ys)] = pis
        else:
            W = np.zeros((self.X, self.Y), dtype=np.float32)

        
        chosen_i_y = np.zeros((self.I, self.Y), dtype=bool)
        for i, S in self.Y_i_y.items():
            if S:
                chosen_i_y[i, list(S)] = True

        chosen_x_j = np.zeros((self.X, self.J), dtype=bool)
        for j, S in self.X_j_x.items():
            if S:
                chosen_x_j[list(S), j] = True

        rc_i = self.phi_i_y.astype(np.float32) - W[self.x_i, :] - u[:, None]
        rc_i[chosen_i_y] = -np.inf  # Block columns already in RMP

        rc_j = self.psi_x_j.astype(np.float32) + W[:, self.y_j] - v[None, :]
        rc_j[chosen_x_j] = -np.inf

        new_columns_i_y, new_columns_j_x = [], []

        # Pick best (or top-k) 
        if topk == 1:
            best_y = rc_i.argmax(axis=1)
            best_rc_i = rc_i[np.arange(self.I), best_y]
            add_i = np.where(best_rc_i > rc_tol)[0]
            for i in add_i:
                y = int(best_y[i])
                self.Y_i_y[i].add(y)
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
                    self.Y_i_y[i].add(int(y))
                    new_columns_i_y.append((int(i), int(y)))
                    seen[i] += 1

        if topk == 1:
            best_x = rc_j.argmax(axis=0)
            best_rc_j = rc_j[best_x, np.arange(self.J)]
            add_j = np.where(best_rc_j > rc_tol)[0]
            for j in add_j:
                x = int(best_x[j])
                self.X_j_x[j].add(x)
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
                    self.X_j_x[int(j)].add(int(x))
                    new_columns_j_x.append((int(j), int(x)))
                    seen[j] += 1

        return new_columns_i_y, new_columns_j_x


    def update_rmp(self, new_columns_i_y, new_columns_j_x):
        for (i, y) in new_columns_i_y:
            x = self.x_i[i].item()
            col = grb.Column()
            col.addTerms(1.0, self.row_i[i])
            col.addTerms(1.0, self.linking_x_y[x,y])
            self.lambda_i_y[i,y] = self.m.addVar(obj=self.phi_i_y[i,y], lb=0.0, column=col, name=f"L_{i}_{y}")
        for (j, x) in new_columns_j_x:
            y = self.y_j[j].item()
            col = grb.Column()
            col.addTerms(1.0, self.row_j[j])
            col.addTerms(-1.0, self.linking_x_y[x,y])
            self.lambda_x_j[x,j] = self.m.addVar(obj=self.psi_x_j[x,j], lb=0.0, column=col, name=f"R_{x}_{j}")
        return
    

    def column_generation(self, max_iter=100, rc_tol=1e-6, verbose=0):
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
        print(f"Iter 0: obj = {self.m.ObjVal:.6f}  (initial BFS)")

        self.m.setParam("Presolve", 0)
        self.m.Params.LPWarmStart = 2

        for k in range(max_iter):
            new_columns_i_y, new_columns_j_x = self.find_improved_reduced_cost(rc_tol=rc_tol)

            if not new_columns_i_y and not new_columns_j_x:     # stop condition
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
        total_vars = int(self.I + self.J + sum(len(s) for s in self.X_j_x.values()) + sum(len(s) for s in self.Y_i_y.values()))

        return history, total_time, total_vars, total_lp_iterations, build_time, self.m


# Dantzig_Wolfe_Estimation class: Reverse problem, dictionary approach

class Dantzig_Wolfe_Estimation:

    def __init__(self, mu_x_y, pi_i_y, pi_x_j, phi_x_y_k, x_i, y_j, eps_i_0=None, eta_0_j=None, eps_i_y=None, eta_x_j=None, seed=0):
        self.I, self.J = len(x_i), len(y_j)
        self.X, self.Y, self.K = phi_x_y_k.shape

        self.mu_x_y = mu_x_y
        self.pi_i_y, self.pi_x_j = pi_i_y, pi_x_j

        self.phi_x_y_k = phi_x_y_k
        self.x_i, self.y_j = x_i, y_j
        self.delta_i_x = np.eye(self.X)[x_i]
        self.delta_j_y = np.eye(self.Y)[y_j]

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

        self.Y_i_y = {i: set() for i in range(self.I)}  # None for singles?
        self.X_j_x = {j: set() for j in range(self.J)}


    def unrestricted_solve(self, verbose=0):
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
        print("\nTotal time:", t)

        return lambda_k, t, build_time, m

  
    def basic_feasible_solution(self):
        idx_i_by_x = [list(np.where(self.x_i == x)[0]) for x in range(self.X)]
        idx_j_by_y = [list(np.where(self.y_j == y)[0]) for y in range(self.Y)]

        for x in range(self.X):
            for y in range(self.Y):
                c = int(self.mu_x_y[x, y])
                for _ in range(c):
                    i = idx_i_by_x[x].pop()
                    j = idx_j_by_y[y].pop()
                    self.Y_i_y[i].add(y)
                    self.X_j_x[j].add(x)
                    
        for x in range(self.X):
            for i in idx_i_by_x[x]:
                self.Y_i_y[i].add(None)
        for y in range(self.Y):
            for j in idx_j_by_y[y]:
                self.X_j_x[j].add(None)
                
        counter = self.K + self.X*self.Y
        i = 0
        while counter > 0 and i < self.I:
            if None not in self.Y_i_y[i]:
                self.Y_i_y[i].add(None)
                counter -= 1
            i += 1
        j = 0
        while counter > 0 and j < self.J:
            if None not in self.X_j_x[j]:
                self.X_j_x[j].add(None)
                counter -= 1
            j += 1
        if counter > 0:
            print(f"Not enough variables to activate.")
        return

  
    def build_rmp(self, verbose=0):
        m = grb.Model("RMP")
        m.Params.OutputFlag = verbose > 0
        m.Params.UpdateMode = 1

        m.setObjective(0, grb.GRB.MAXIMIZE)

        row_i, row_j, linking_x_y, linking_k = {}, {}, {}, {}

        lambda_i_0, lambda_i_y = {}, {}
        for i in range(self.I):  # individual constraints i
            if len(self.Y_i_y[i]) == 1:
                y = next(iter(self.Y_i_y[i]))
                if y is None:
                    lambda_i_0[i] = m.addVar(obj=self.eps_i_0[i], lb=0.0)
                    row_i[i] = m.addConstr(lambda_i_0[i] == 1.0, name=f"row_i_{i}")
                else:
                    lambda_i_y[i, y] = m.addVar(obj=self.eps_i_y[i, y], lb=0.0, name=f"L_{i}_{y}")
                    row_i[i] = m.addConstr(lambda_i_y[i, y] == 1.0, name=f"row_i_{i}")
            else:  # in this case None is necessarily in the set
                lambda_i_0[i] = m.addVar(obj=self.eps_i_0[i], lb=0.0)
                row_i[i] = m.addConstr(lambda_i_0[i] == 1.0, name=f"row_i_{i}")
                y = next(iter(self.Y_i_y[i] - set({None})))
                col = grb.Column()
                col.addTerms(1.0, row_i[i])
                lambda_i_y[i, y] = m.addVar(obj=self.eps_i_y[i, y], lb=0.0, column=col, name=f"L_{i}_{y}")

        lambda_0_j, lambda_x_j = {}, {}
        for j in range(self.J):  # individual constraints j
            if len(self.X_j_x[j]) == 1:
                x = next(iter(self.X_j_x[j]))
                if x is None:
                    lambda_0_j[j] = m.addVar(obj=self.eta_0_j[j], lb=0.0)
                    row_j[j] = m.addConstr(lambda_0_j[j] == 1.0, name=f"row_j_{j}")
                else:
                    lambda_x_j[x, j] = m.addVar(obj=self.eta_x_j[x, j], lb=0.0, name=f"R_{x}_{j}")
                    row_j[j] = m.addConstr(lambda_x_j[x, j] == 1.0, name=f"row_j_{j}")
            else:  # in this case None is necessarily in the set
                lambda_0_j[j] = m.addVar(obj=self.eta_0_j[j], lb=0.0)
                row_j[j] = m.addConstr(lambda_0_j[j] == 1.0, name=f"row_j_{j}")
                x = next(iter(self.X_j_x[j] - set({None})))
                col = grb.Column()
                col.addTerms(1.0, row_j[j])
                lambda_x_j[x, j] = m.addVar(obj=self.eta_x_j[x, j], lb=0.0, column=col, name=f"R_{x}_{j}")

        for x in range(self.X):
            for y in range(self.Y):
                lhs = grb.LinExpr()
                for i in range(self.I):
                    if self.x_i[i] == x and (i, y) in lambda_i_y:
                        lhs += lambda_i_y[i, y]
                for j in range(self.J):
                    if self.y_j[j] == y and (x, j) in lambda_x_j:
                        lhs -= lambda_x_j[x, j]
                linking_x_y[x, y] = m.addConstr(lhs == 0.0, name=f"linking_{x}_{y}")

        self.linking_k_list = []
        for k in range(self.K):  # linking constraints k
            lhs = grb.LinExpr()
            for i in range(self.I):
                ys = [y for y in self.Y_i_y[i] if y is not None]
                if len(ys) == 1:
                    y = ys[0]
                    lhs += lambda_i_y[i, y] * self.phi_x_y_k[self.x_i[i], y, k] / 2
            for j in range(self.J):
                xs = [x for x in self.X_j_x[j] if x is not None]
                if len(xs) == 1:
                    x = xs[0]
                    lhs += lambda_x_j[x, j] * self.phi_x_y_k[x, self.y_j[j], k] / 2
            rhs = ((self.phi_x_y_k * self.mu_x_y[:, :, None]).sum(axis=(0, 1)))[k]
            constr = m.addConstr(lhs == rhs)
            linking_k[k] = constr
            self.linking_k_list.append(constr)

        self.lambda_i_0, self.lambda_i_y = lambda_i_0, lambda_i_y
        self.lambda_0_j, self.lambda_x_j = lambda_0_j, lambda_x_j
        self.row_i, self.row_j, self.linking_x_y, self.linking_k = row_i, row_j, linking_x_y, linking_k
        self.m = m

        return m

  
    def get_lambdas(self):
        return -np.array([constr.Pi for constr in self.linking_k_list])

  
    def find_improved_reduced_cost(self, rc_tol=1e-6, verbose=0):
        u_i = {i: self.row_i[i].Pi for i in range(self.I)}
        v_j = {j: self.row_j[j].Pi for j in range(self.J)}
        W_x_y = np.zeros((self.X, self.Y)) # matrix access is faster than dictionary for this size
        for a in self.linking_x_y:
            W_x_y[a] = self.linking_x_y[a].Pi
        lambda_k = -np.array([self.linking_k[k].Pi for k in range(self.K)])

        new_columns_i_y, new_columns_j_x = [], []

        Phi_lambda_x_y = self.phi_x_y_k @ lambda_k

        Y_i_y_cached = self.Y_i_y
        U_i_y_cached = self.delta_i_x @ (Phi_lambda_x_y / 2 - W_x_y) + self.eps_i_y
        for i in range(self.I):
            x = self.x_i[i].item()
            Y_i = np.ones(self.Y, dtype=bool)
            single_flag = True
            for y in Y_i_y_cached[i]:
                if y is None:
                    single_flag = False
                else:
                    Y_i[y] = False
            best_rc = -np.inf
            best_y = None
            if Y_i.any() > 0:
                rc_y = U_i_y_cached[i,Y_i] - u_i[i]
                best_idx = np.argmax(rc_y)
                best_rc = rc_y[best_idx]
                best_y = np.arange(self.Y)[Y_i][best_idx]
            if single_flag:
                rc_0 = self.eps_i_0[i] - u_i[i]
                if rc_0 > best_rc:
                    best_rc = rc_0
                    best_y = None
            if best_rc > rc_tol:
                self.Y_i_y[i].add(best_y)
                new_columns_i_y.append((i,best_y))
                if verbose > 1: print(f"Type y={best_y} entered choice set of i={i}. Reduced cost: {best_rc:.2f}.")

        X_j_x_cached = self.X_j_x
        V_x_j_cached = (Phi_lambda_x_y / 2 + W_x_y) @ self.delta_j_y.T + self.eta_x_j
        for j in range(self.J):
            y = self.y_j[j].item()
            X_j = np.ones(self.X, dtype=bool)
            single_flag = True
            for x in X_j_x_cached[j]:
                if x is None:
                    single_flag = False
                else:
                    X_j[x] = False
            best_rc = -np.inf
            best_x = None
            if X_j.any() > 0:
                rc_x = V_x_j_cached[X_j,j] - v_j[j]
                best_idx = np.argmax(rc_x)
                best_rc = rc_x[best_idx]
                best_x = np.arange(self.X)[X_j][best_idx]
            if single_flag:
                rc_0 = self.eta_0_j[j] - v_j[j]
                if rc_0 > best_rc:
                    best_rc = rc_0
                    best_x = None
            if best_rc > rc_tol:
                self.X_j_x[j].add(best_x)
                new_columns_j_x.append((j,best_x))
                if verbose > 1: print(f"Type x={best_x} entered choice set of j={j}. Reduced cost: {best_rc:.2f}.")

        return new_columns_i_y, new_columns_j_x

  
    def update_rmp(self, new_columns_i_y, new_columns_j_x):
        for (i, y) in new_columns_i_y:
            if y is None:
                col = grb.Column()
                col.addTerms(1.0, self.row_i[i])
                self.lambda_i_0[i] = self.m.addVar(obj=self.eps_i_0[i], lb=0.0, column=col)
            else:
                x = int(self.x_i[i])
                col = grb.Column()
                col.addTerms(1.0, self.row_i[i])
                col.addTerms(1.0, self.linking_x_y[x,y])
                for k in range(self.K):
                    col.addTerms(self.phi_x_y_k[x,y,k]/2, self.linking_k[k])
                self.lambda_i_y[i,y] = self.m.addVar(obj=self.eps_i_y[i,y], lb=0.0, column=col, name=f"L_{i}_{y}")
        for (j, x) in new_columns_j_x:
            if x is None:
                col = grb.Column()
                col.addTerms(1.0, self.row_j[j])
                self.lambda_0_j[j] = self.m.addVar(obj=self.eta_0_j[j], lb=0.0, column=col)
            else:
                y = int(self.y_j[j])
                col = grb.Column()
                col.addTerms(1.0, self.row_j[j])
                col.addTerms(-1.0, self.linking_x_y[x,y])
                for k in range(self.K):
                    col.addTerms(self.phi_x_y_k[x,y,k]/2, self.linking_k[k])
                self.lambda_x_j[x,j] = self.m.addVar(obj=self.eta_x_j[x,j], lb=0.0, column=col, name=f"R_{x}_{j}")
        return

  
    def column_generation(self, max_iter=100, rc_tol=1e-6):
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
        print(f"Iter 0: obj = {self.m.ObjVal:.6f}  (initial BFS)")

        self.m.setParam("Presolve", 0)
        self.m.Params.LPWarmStart = 2

        for iter in range(max_iter):
            new_columns_i_y, new_columns_j_x = self.find_improved_reduced_cost(rc_tol=rc_tol)

            if not new_columns_i_y and not new_columns_j_x:  # stop condition
                print(f"Iter {iter + 1:2d}: no positive reduced cost – optimal.")
                break

            self.update_rmp(new_columns_i_y, new_columns_j_x)

            self.m.optimize()
            print("====")
            print(self.m.ObjVal)
            total_lp_iterations += self.m.IterCount  # Count iterations
            history.append(self.m.ObjVal)

        total_time = time.perf_counter() - start_time
        total_vars = int(sum(len(s) for s in self.X_j_x.values()) + sum(len(s) for s in self.Y_i_y.values()))

        return history, total_time, total_vars, total_lp_iterations, build_time


# Visualization tools

def visualize_results(unr_list, dw_list):
    _, _, t_unr, build_unr, m_unr = unr_list
    
    history_dict, dw_time, dw_vars, dw_iterations, build_dw, dw_model = dw_list
    dw_obj = history_dict[-1] if history_dict else 0
    
    if m_unr.Status == grb.GRB.OPTIMAL:
        obj_unr = m_unr.ObjVal
        obj_diff = abs(obj_unr - dw_obj)
        print(f"Objective values: BF={obj_unr:.6f}, DW={dw_obj:.6f}")
        print(f"\nMetrics comparison (from feedback):")

        print(f"{'Metric':<25} {'Full-column LP':<20} {'Dantzig-Wolfe':<20} {'Ratio/Speedup':<15}")
        print("-" * 80)
        print(f"{'Build time (s)':<25} {build_unr:<20.3f} {build_dw:<20.3f} {build_unr/build_dw:<15.2f}")
        print(f"{'Solve time (s)':<25} {t_unr:<20.3f} {dw_time:<20.3f} {t_unr/dw_time:<15.2f}")

        print(f"{'Iter (LP iterations)':<25} {int(m_unr.IterCount):<20d} {int(dw_iterations):<20d} {int(m_unr.IterCount)/int(dw_iterations):<15}")
        print(f"{'N_col (variables)':<25} {m_unr.NumVars:<20d} {dw_vars:<20d} {m_unr.NumVars/dw_vars:<15.1f}")
        if obj_diff < 1e-6:
            print(f"{'Obj (optimal value)':<25} {obj_unr:<20.6f} {dw_obj:<20.6f} {'Match':<15}")
        else:
            print(f"{'Obj (optimal value)':<25} {obj_unr:<20.6f} {dw_obj:<20.6f} {'Mismatch':<15}")
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
        s.plot(title="Dantzig–Wolfe Convergence", xlabel="Iteration", ylabel="Objective Value", marker='o', linewidth=2)
        plt.axhline(y=obj_unr, color='r', linestyle='--', label='Optimal (Full-column LP)', alpha=0.7)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
