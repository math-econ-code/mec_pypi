import numpy as np
import gurobipy as grb
import matplotlib.pyplot as plt


class Polyhedral():
    def __init__(self,y_t_k,v_t,ytilde_s_k= np.array([]),vtilde_s = np.array([]),namef = '',namesv = None,verbose=0):
        if namesv is None:
            namesv = 'x'
        self.nbt,self.nbk= y_t_k.shape
        self.namef = namef
        if type(namesv)==str:
            if self.nbk==1:
                self.namesv = [namesv[0]] 
            else:
                self.namesv = [namesv[0]+'_'+str(k) for k in range(self.nbk)] 
        elif (type(namesv)==list) & (len(namesv)==self.nbk):
            self.namesv = namesv
        else:
            raise Exception("Parameter namesv not provided under the right format.")
        if ytilde_s_k.shape == (0,):
            self.nbs = 0
            ytilde_s_k = ytilde_s_k.reshape((0,self.nbk))
        else:
            self.nbs = ytilde_s_k.shape[0]
        self.nbi = self.nbt+self.nbs
        self.nbj = self.nbi+self.nbk+1
        if self.nbk > self.nbi:
            print('Caution: dimension larger than number of constraints.')
        self.ytilde_s_k = ytilde_s_k
        self.y_t_k = y_t_k
        self.vtilde_s = vtilde_s.reshape(self.nbs)
        self.v_t = v_t.reshape(self.nbt)
        self.tableau_i_j = np.block([[self.y_t_k, -np.ones((self.nbt,1)), np.eye(self.nbt),np.zeros( (self.nbt,self.nbs) ) ],
                                     [self.ytilde_s_k, -np.zeros((self.nbs,1)), np.zeros( (self.nbs,self.nbt) ), np.eye(self.nbs) ]])
        self.rhs_i = np.concatenate([self.v_t,self.vtilde_s])
        j_n = list(range(self.nbk+1))
        m = grb.Model()
        m.setParam('OutputFlag', 0)
        x_j = m.addMVar(self.nbj, lb = (self.nbk+1)*[-grb.GRB.INFINITY]+self.nbi*[0])
        m.setObjective( x_j[:self.nbk]@(- self.y_t_k[0,:]) + x_j[self.nbk], sense = grb.GRB.MINIMIZE)
        m.addConstr(self.tableau_i_j @ x_j == self.rhs_i)
        m.optimize()
        if m.Status in (grb.GRB.INFEASIBLE,grb.GRB.INF_OR_UNBD):
            self.all_infinite = 1
            if verbose>0:
                print('Empty domain')
        else:
            self.all_infinite = 0
            self.j_n = [i for  (i,v) in enumerate(m.getVars() ) if v.vBasis == -1]
            if verbose>0:
                print('Initial nonbasic columns=',self.j_n)


    def val(self,x_k):
        if np.array(x_k).shape ==():
            x_k = np.array([x_k])
        if self.nbs > 0:
            if (self.ytilde_s_k @ x_k - self.vtilde_s).max()>0:
                return float('inf')
        return (self.y_t_k @ x_k - self.v_t).max()

    def grad(self,x_k):
        if np.array(x_k).shape ==():
            x_k = np.array([x_k])
        if self.nbs > 0:
            if (self.ytilde_s_k @ x_k - self.vtilde_s).max()>0:
                return float('inf')
        k = (self.y_t_k @ x_k - self.v_t).argmax()
        return self.y_t_k[:,k]

    def __repr__(self,num_digits = 2,with_name = True):
        if self.all_infinite == 1:
            return '+ infinity everywhere' 
        elif self.all_infinite == -1:
            return '- infinity everywhere' 
        else:
            from sympy import Symbol
            from mec.lp import round_expr
            x_k = [Symbol(namev) for namev in self.namesv]
            if self.nbt >1:
                obj = f'max{str({round_expr(e,num_digits) for e in list(self.y_t_k @ x_k - self.v_t)} )}'
            else:
                obj = str( (self.y_t_k @ x_k - self.v_t)[0])
            constrs = [f'{round_expr(self.ytilde_s_k[s,:] @ x_k,num_digits)} <= {round(self.vtilde_s[s],num_digits)}' for s in range(self.nbs)]
            vars_str = ''.join([ name+ (',' if i<self.nbk-1 else '') for (i,name) in enumerate(self.namesv) ])
            vars_str_with_parenthesis = ('(' if self.nbk>1 else '') + vars_str + (')' if self.nbk>1 else '') 
            if with_name:
                if self.namef == '':
                    result = 'Function: '
                else:
                    result = self.namef + '(' + vars_str + ')' 
            else:
                result = ''
            result += str(obj)
            if (self.nbk==1) and (self.nbs>0):
                result += '\n'+ ('Domain: '+ self.namesv[0]+f' in {self.domain1d(num_digits)}')
            elif (self.nbk>1) and (self.nbs>0):
                result += '\n'+ 'Domain: ' + vars_str_with_parenthesis +' s.t. ' +('\n' if self.nbs>1 else '')
                for c in constrs:
                    result +=  str(c)+ '\n'
            else: # if there are no constraints
                result += '\n'+ str('Domain: ' + vars_str_with_parenthesis + ' in R' + ('^'+str(self.nbk) if self.nbk>1 else '') +'.\n')
            return result

    def domain1d(self,num_digits = 2):
        xl = max([float('-inf')]+[self.vtilde_s[s] / self.ytilde_s_k[s] for s in range(self.nbs) if self.ytilde_s_k[s]<0])
        xu = min([float('inf')]+[self.vtilde_s[s] / self.ytilde_s_k[s] for s in range(self.nbs) if self.ytilde_s_k[s]>0])
        return(round(float(xl),2), round(float(xu),2))

    def plot1d(self, xl=-10,xu=10,verbose = 0):
        if self.nbk >1:
            print('not 1d.')
        xla = max([float('-inf')]+[self.vtilde_s[s] / self.ytilde_s_k[s] for s in range(self.nbs) if self.ytilde_s_k[s]<0])
        xua = min([float('inf')]+[self.vtilde_s[s] / self.ytilde_s_k[s] for s in range(self.nbs) if self.ytilde_s_k[s]>0])
        if verbose>0:
            print(f'Domain=({float(xla)},{float(xua)})')
        xl = max(xla,xl)
        xu = min(xua,xu)
        xs = np.linspace(xl, xu, 400)
        ys = [self.val(x) for x in xs]
        plt.plot(xs, ys, label=self.namef+f'({self.namesv[0]})')
        plt.xlabel(self.namesv[0])
        plt.ylabel(self.namef)
        plt.legend()
        plt.show()


    def j_b(self,j_n = None):
        if j_n is None:
            j_n = self.j_n
        return [j for j in range(self.nbj) if j not in j_n]


    def subtableau_i_b(self,j_n=None):
        return self.tableau_i_j[:,self.j_b(j_n) ]

    def subtableau_i_n(self, j_n=None):
        if j_n is None:
            j_n = self.j_n
        return self.tableau_i_j[:,j_n ]

    def basic_solution_i(self, j_n):
        return np.linalg.solve(self.subtableau_i_b(j_n),self.rhs_i)

    def basic_infinite_solution_i(self,j_n,jent):
        ient = jent - self.nbk - 1
        therhs_i = np.zeros(self.nbi)
        therhs_i[ient] = -1
        return np.linalg.solve(self.subtableau_i_b(j_n),therhs_i)


    def dictionary(self,j_n=None): # of the form x_b = sol_b - D_b_n @ x_n
        sol_b = np.linalg.solve(self.subtableau_i_b(j_n),self.rhs_i)
        D_b_n = np.linalg.solve(self.subtableau_i_b(j_n),self.subtableau_i_n(j_n))
        return (sol_b,D_b_n)

    def determine_departing(self,j_n,jent):
        nent = [n for (n,j) in enumerate(j_n) if j==jent][0]
        (sol_b,D_b_n) = self.dictionary(j_n)
        D_b = D_b_n[:,nent]
        thedic = {b: sol_b[b] / D_b[b]
                  for b in range(self.nbk+1,self.nbi) # the nbk+1 first basic variables are the x_k and u and are unconstrained 
                  if  D_b[b] >0}
        if len (thedic)==0:
            return -1
        else:
            bdep = min(thedic, key = thedic.get)
            return self.j_b(j_n)[bdep]

    def star(self, namesv = None):
        if self.all_infinite != 0:
            v = Polyhedral(np.zeros( (1,self.nbk) ) ,np.zeros(1),namesv = namesv )
            v.all_infinite = - self.all_infinite 
            return v
        import networkx as nx
        the_graph = nx.DiGraph()
        j_n =  self.j_n.copy()
        xu = self.basic_solution_i(j_n)[:(self.nbk+1)]
        the_dict = {frozenset(j_n): xu }
        labels_to_add = [j_n]
        the_graph.add_nodes_from([frozenset(j_n)] )
        while len(labels_to_add)>0:
            j_n = labels_to_add.pop()
            for jent in j_n:
                jdep = self.determine_departing(j_n,jent)
                jnext_n = list({jdep} | set(j_n) -  {jent})
                if jdep > -1: # the node jnext_n is central
                    if frozenset(jnext_n) not in the_dict.keys():
                        labels_to_add.append(jnext_n) # attach to labels_to_add
                        the_graph.add_edges_from([(frozenset(j_n),frozenset(jnext_n)) ]) # add to the graph
                        xu = self.basic_solution_i(jnext_n)[:(self.nbk+1)] # find info 
                        the_dict.update({frozenset(jnext_n): xu }) # add to the dictionary
                else: #jdep == -1 ; means the node is exterior
                    # do not attach to labels_to_add
                    the_graph.add_edges_from([(frozenset(j_n),frozenset(jnext_n)) ]) # add to the graph
                    xu = self.basic_infinite_solution_i(j_n,jent)[:(self.nbk+1)] # find info 
                    the_dict.update({frozenset(jnext_n): xu }) # add to the dictionary

        xtilde_list = []
        utilde_list = []
        x_list = []
        u_list = []
        for f in the_dict.keys():
            if -1 in f:
                (x_k,u) = (the_dict[f][:-1],the_dict[f][-1])
                if np.abs(x_k).sum()>0: # only attach if x_k is not the zero vector
                    xtilde_list.append(x_k)
                    utilde_list.append(u)
            else:
                x_list.append(the_dict[f][:-1])
                u_list.append(the_dict[f][-1])
        xtilde_m_k = np.array(xtilde_list)
        utilde_m = np.array(utilde_list)
        x_m_k =  np.array(x_list)
        u_m = np.array(u_list)
        
        if namesv is None:
            namesv = [ (chr(ord(name[0])+1)+name[1:]) for name in self.namesv]
        namefstar = (self.namef+'*' if len(self.namef) >0 else '')
        ustar = Polyhedral(x_m_k,u_m,xtilde_m_k,utilde_m,namef=namefstar,namesv=namesv )

        return (ustar)
        
        
    def __add__(self,u2):
        u1 = self
        if isinstance(u2,Polyhedral): # addition of two polyhedral functions
            if u1.nbk == u2.nbk:
                nbk = u1.nbk
            else:
                print('Dimensions do not match.')
            y_t1t2_k = (u1.y_t_k[:,None,:] + u2.y_t_k[None,:,:]).reshape((-1,nbk))
            v_t1t2 = (u1.y_t_k[:,None] + u2.y_t_k[None,:]).flatten()
            ytilde_s_k = np.block([[u1.ytilde_s_k],[u2.ytilde_s_k]])
            vtilde_s = np.concatenate([u1.vtilde_s,u2.vtilde_s])
            if (u1.namesv == u2.namesv):
                namesv = u1.namesv
            else:
                namesv = None
            usum = Polyhedral(y_t1t2_k, v_t1t2,ytilde_s_k,vtilde_s, namesv = namesv )
            return(usum)
            
        elif isinstance(u2,(int,float)): # addition of a polyhedral function and a scalar
            return Polyhedral(u1.y_t_k,u1.v_t - u2,u1.ytilde_s_k,u1.vtilde_s, namesv = u1.namesv)
        
        else:
            raise NotImplementedError("Can't add Polyhedral and {}".format(type(u2)))
        
    def __radd__(self,scal):
        return self.__add__(scal)
    
    def __sub__(self,scal):
        if isinstance(scal,(int,float)): 
            return Polyhedral(self.y_t_k,self.v_t + scal,self.ytilde_s_k,self.vtilde_s, namesv = self.namesv)
        else:
            raise NotImplementedError("Can't substract Polyhedral from {}".format(type(scal)))
    
    def __mul__(self,scal):
        if isinstance(scal,(int,float)): 
            return Polyhedral(scal* self.y_t_k,scal* self.v_t,self.ytilde_s_k,self.vtilde_s, namesv = self.namesv)
        else:
            raise NotImplementedError("Can't multiply Polyhedral and {}".format(type(scal)))
            
    def __rmul__(self,scal):
        return self.__mul__(scal)
        
    def __truediv__(self,scal):
        if isinstance(scal,(int,float)): 
            return Polyhedral( self.y_t_k / scal, self.v_t / scal,self.ytilde_s_k,self.vtilde_s, namesv = self.namesv)
        else:
            raise NotImplementedError("Can't divide {} by Polyhedral.".format(type(scal)))


    def __or__(self,u2): 
        u1 = self
        if isinstance(u2,Polyhedral): # maximum of two PHC functions
            if u1.nbk == u2.nbk:
                nbk = u1.nbk
            else:
                print('Dimensions do not match.')
            y_t_k = np.block([[u1.y_t_k],[u2.y_t_k]])
            v_t = np.concatenate([u1.v_t,u2.v_t])
            ytilde_s_k = np.block([[u1.ytilde_s_k],[u2.ytilde_s_k]])
            vtilde_s = np.concatenate([u1.vtilde_s,u2.vtilde_s])
            if (u1.namesv == u2.namesv):
                namesv = u1.namesv
            else:
                namesv = None
            umax = Polyhedral(y_t_k, v_t,ytilde_s_k,vtilde_s, namesv = namesv )
            return(umax)
            
        elif isinstance(u2,(int,float)): # addition of a polyhedral function and a scalar
            return u1.__or__(Polyhedral(np.zero((1,u1.nbk), np.array([-u2]) , namesv = u1.namesv )))
        
        else:
            raise NotImplementedError("Can't take the maximum of Polyhedral and {}".format(type(u2)))
        
    def __ror__(self,scal):
        return self.__or__(scal)


    def __xor__(self,u2): # inf-convolution of two PHC functions
        if isinstance(u2,Polyhedral):
            if self.nbk == u2.nbk:
                nbk = self.nbk
            else:
                print('Dimensions do not match.')
            return (self.star() + u2.star()).star()
            
        else:
            raise NotImplementedError("Can't take the inf-convolution of Polyhedral and {}".format(type(u2)))
    
    
    # def graph(self):
        # # build skeleton:
        # import networkx as nx
        # the_graph = nx.DiGraph()
        # j_n =  self.j_n.copy()
        # xu = self.basic_solution_i(j_n)[:(self.nbk+1)]
        # the_dict = {frozenset(j_n): xu }
        # labels_to_add = [j_n]
        # the_graph.add_nodes_from([frozenset(j_n)] )
        # while len(labels_to_add)>0:
            # j_n = labels_to_add.pop()
            # for jent in j_n:
                # jdep = self.determine_departing(j_n,jent)
                # jnext_n = list({jdep} | set(j_n) -  {jent})
                # if jdep > -1: # the node jnext_n is central
                    # if frozenset(jnext_n) not in the_dict.keys():
                        # labels_to_add.append(jnext_n) # attach to labels_to_add
                        # the_graph.add_edges_from([(frozenset(j_n),frozenset(jnext_n)) ]) # add to the graph
                        # xu = self.basic_solution_i(jnext_n)[:(self.nbk+1)] # find info 
                        # the_dict.update({frozenset(jnext_n): xu }) # add to the dictionary
                # else: #jdep == -1 ; means the node is exterior
                    # # do not attach to labels_to_add
                    # the_graph.add_edges_from([(frozenset(j_n),frozenset(jnext_n)) ]) # add to the graph
                    # xu = self.basic_infinite_solution_i(j_n,jent)[:(self.nbk+1)] # find info 
                    # the_dict.update({frozenset(jnext_n): xu }) # add to the dictionary
        
        # #### HERE: something to build the bipartite graph            


            
def polyhedral_from_strings(expr_fun_str ,expr_dom_strs = [], verbose= 0):
    # for example exammple: 
    # expr_fun_str = 'max(3*a+2*b-1, 4*a-b+3,7*a-3*b+9)'
    # expr_dom_strs = ['a+1 <= 0 ','b >= 1']
    import sympy
    expr_fun = sympy.sympify(expr_fun_str)
    expr_doms = [sympy.sympify(expr_dom_str) for expr_dom_str in expr_dom_strs]
    variables = sorted(expr_fun.free_symbols.union(*[expr_dom.free_symbols for expr_dom in expr_doms] ), key=lambda x: x.name)
    if verbose:
        print('Variables =' , variables)
    list_y = []
    list_v = []
    for expr in expr_fun.args:
        coeffs = expr.as_coefficients_dict() 
        list_y.append([float(coeffs.get(v,0)) for v in variables] )
        list_v.append(-float(coeffs.get(1,0)))
    y_t_k = np.array(list_y)
    v_t = np.array(list_v)
    if len(expr_dom_strs) == 0:
        return Polyhedral(y_t_k,v_t) 

    list_ytilde = []
    list_vtilde = []

    for expr in expr_doms:
        lhs, rhs = expr.args
        if (expr.func == sympy.core.relational.LessThan) or (expr.func == sympy.core.relational.StrictLessThan):
            diff = lhs - rhs
        elif (expr.func == sympy.core.relational.GreaterThan) or (expr.func == sympy.core.relational.StrictGreaterThan):
            diff = rhs - lhs
        else:
            print('Not expected format.')
        coeffs = diff.as_coefficients_dict()
        list_ytilde.append([float(coeffs.get(v,0)) for v in variables] )
        list_vtilde.append(-float(coeffs.get(1,0)))
    ytilde_s_k = np.array(list_ytilde)
    vtilde_s = np.array(list_vtilde)
    return Polyhedral(y_t_k,v_t,ytilde_s_k,vtilde_s,namesv = [str(var) for var in variables] )


    # def plot2d

    # def relational_graph
    
    # soft max regularization

    # Hessian and possibly higher order derivatives
    
    # power diagrams 
