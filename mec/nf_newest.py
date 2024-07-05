# Network flow problems library
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from mec.lp import Tableau, two_phase
from anytree import Node, RenderTree

def create_connected_dag(num_nodes, num_edges, zero_node = 0, seed=777):
    np.random.seed(seed)
      
    cont = True
    while cont:
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        while len(G.edges) < num_edges:
            a, b = np.random.choice(G.nodes, 2,replace=False)
            if not nx.has_path(G, b, a):
                G.add_edge(a, b)

        if not nx.is_weakly_connected(G):
            components = list(nx.weakly_connected_components(G))
            for i in range(len(components) - 1):
                a = np.random.choice(components[i])
                b = np.random.choice(components[i+1])
                G.add_edge(a, b)

        node_list = [ 'z'+str(node) for node in G.nodes]
        arcs_list = [(node_list[i],node_list[j]) for i,j in G.edges]
        basis = list(np.random.choice(list( range(num_edges)), num_nodes-1 ) )
        mu_a = np.zeros(num_edges)
        mu_a[basis] = np.random.randint(0, num_edges, num_nodes - 1 )
        c_a = np.random.randint(0, num_edges, num_edges)

        q_z = np.array(nx.incidence_matrix(G, oriented=True).todense() @ mu_a)
        if q_z[0] <0:
            cont= False

    return (node_list,arcs_list,c_a,q_z,G)



def create_connected_bipartite(nbx, nby,  zero_node = 0, seed=777):
    num_nodes = nbx + nby
    num_edges = nbx * nby
    np.random.seed(seed)
    cont = True
    G = nx.DiGraph()
    G.add_nodes_from(['x'+str(i) for i in range(nbx)], bipartite= 0)
    G.add_nodes_from(['y'+str(j) for j in range(nby)], bipartite= 1)
    G.add_edges_from([('x'+str(i),'y'+str(j)) for j in range(nby)  for i in range(nbx)] )

    
    node_list = list(G.nodes)
    arcs_list = list(G.edges)
    basis = list(np.random.choice(list( range(num_edges)), num_nodes-1 ) )
    mu_a = np.zeros(num_edges)
    mu_a[basis] = np.random.randint(0, num_edges, num_nodes - 1 )+1
    c_a = np.random.randint(0, num_edges, num_edges)+1

    q_z = np.array(nx.incidence_matrix(G, oriented=True).todense() @ mu_a)
    #if q_z[0] <0:
    #    cont= False
    return (node_list,arcs_list,c_a,q_z,G)


class Network_problem:
    def __init__(self, nodesList, arcsList, c_a, q_z, active_basis=None, zero_node=0, pos=None, seed=777, verbose=0):
        self.zero_node = zero_node
        self.nbz = len(nodesList)
        self.nba = len(arcsList)
        self.nodesList = nodesList
        self.arcsList = arcsList
        self.c_a = c_a
        self.nodesDict = {node:node_ind for (node_ind,node) in enumerate(self.nodesList)}
        self.arcsDict = {arc:arc_ind for (arc_ind,arc) in enumerate(self.arcsList)}
        if verbose>1:
            print('Number of nodes = '+str(self.nbz)+'; number of arcs = '+str(self.nba)+'.')

        data = np.concatenate([-np.ones(self.nba),np.ones(self.nba)])
        arcsIndices = list(range(self.nba))
        arcsOrigins = [self.nodesDict[o] for o,d in self.arcsList]
        arcsDestinations = [self.nodesDict[d] for o,d in self.arcsList]
        
        znotzero = [i for i in range(self.nbz) if i != zero_node]
        self.q_z = q_z
        self.q0_z = q_z[znotzero]
        
        self.nabla_a_z = np.array(sp.csr_matrix((data, (arcsIndices+arcsIndices, arcsOrigins+arcsDestinations)), shape = (self.nba,self.nbz)).todense())
        self.nabla0_a_z = self.nabla_a_z[:,znotzero]
        
        active_basis = two_phase(self.nabla0_a_z.T,self.q0_z)
        assert len(active_basis) == (self.nbz - 1)
        
        self.digraph = nx.DiGraph()
        self.digraph.add_edges_from(arcsList)
        if pos is None:
            pos = nx.spring_layout(self.digraph, seed=seed)
        self.pos = pos

        arcsNames = [str(x)+str(y) for (x,y) in self.arcsList]

        self.tableau = Tableau(A_i_j = np.asarray(np.linalg.solve(self.B(active_basis),self.N(active_basis))),
                               d_i = self.q0_z,
                               c_j = self.gain_a(active_basis)[self.nonbasis(active_basis)],
                               slack_var_names_i = [arcsNames[i] for i in active_basis],
                               decision_var_names_j = [arcsNames[i] for i in self.nonbasis(active_basis)])
        the_arcs_indices = list(active_basis) + list(set(range(self.nba)) - set(active_basis))
        self.a_k = the_arcs_indices
        self.k_a = {the_arcs_indices[k]:k for k in range(self.nba)}

        
        
    
    def draw(self, p_z = None,mu_a = None, gain_a = None, entering_a = None, departing_a = None,figsize=(50, 30)):
        edge_labels = {e:('c='+f"{c:.0f}") for (e,c) in zip(self.arcsList,self.c_a)  }
        if mu_a is not None:
            for (i,e) in enumerate(self.arcsList):
                if i in self.basis(): 
                    edge_labels[e] += '\nμ='+f"{mu_a[i]:.0f}"

        if gain_a is not None:
            for (i,e) in enumerate(self.arcsList):
                if gain_a[i]!=self.c_a[i]: 
                    edge_labels[e] += '\ng='+f"{gain_a[i]:.0f}"

            
        nx.draw_networkx_edge_labels(self.digraph,self.pos,
                                    edge_labels=edge_labels,
                                    font_color='red')
                                    
        nx.draw(self.digraph,self.pos,
                edgelist=[self.arcsList[i] for i  in range(self.nba) if i not in self.basis() ],
                style='dotted', 
                with_labels=False)
                                              
        nx.draw_networkx_edges(self.digraph, self.pos, edgelist=[self.arcsList[i] for i in self.basis()], 
                               edge_color='blue') #connectionstyle='arc3,rad=0.3'
        
        labels = {z: f"q={self.q_z[i]:.0f}"+'\n'+z for i,z in enumerate(self.nodesList) }
        label_pos = {z: (position[0], position[1] ) for z, position in self.pos.items()}        
        if p_z is not None:
            p_z = np.concatenate([np.zeros(1),p_z])
            labels = {z: labels[z]+ f"\np={p_z[i]:.0f}" for i,z in enumerate(self.nodesList)} 
        
        nx.draw_networkx_labels(self.digraph, label_pos, labels,font_size=10,verticalalignment = 'center')
            
        if entering_a is not None:
            nx.draw_networkx_edges(self.digraph, self.pos, edgelist=[self.arcsList[entering_a]], 
                               edge_color='green', connectionstyle='arc3,rad=0.3')
            
        if departing_a is not None:
            nx.draw_networkx_edges(self.digraph, self.pos, edgelist=[self.arcsList[departing_a]], 
                                   style='dotted',
                                   edge_color='white')

        plt.figure(figsize=figsize)
        plt.show()

    # \nabla_0^\top = [B, N] where B corresponds to the basis and N to the non basis

    def basis(self, basis = None):
        if basis is None:
            basis = [ self.a_k[k] for k in self.tableau.k_b]
        return (basis)
    
    def nonbasis(self, basis = None):
        return ([i for i in range(self.nba) if i not in self.basis(basis)])
    
    
    def B(self, basis = None):
        return(self.nabla0_a_z[self.basis(basis),:].T) 
    
    def N(self, basis = None): 
        return(self.nabla0_a_z[self.nonbasis(basis),:].T)
        
    
    def musol_a(self,basis = None):
        mu_a = np.zeros(self.nba)
        mu_a[self.basis(basis)] = np.linalg.solve(self.B(),self.q0_z)
        return mu_a
        
    def p0sol_z(self,basis = None):
        return (np.linalg.solve(self.B(basis).T,self.c_a[self.basis(basis)]))
    
    def gain_a(self, basis = None):
        p_z = self.p0sol_z(basis)
        g_a = self.c_a.copy()
        g_a[self.nonbasis(basis)] = self.N(basis).T @ p_z 
        return g_a
    
    def cost_improvement_a(self, basis = None):
        return( self.gain_a() - self.c_a )
        
    def determine_entering_arcs(self,basis = None):
        return( np.where(self.cost_improvement_a(basis) > 0 )[0].tolist() )
    
    def tableau_update(self,entering_a,departing_a):
        self.tableau.update(self.k_a[entering_a],self.k_a[departing_a])
    
    def determine_departing_arc(self,entering_a, basis=None):
        entering_k = self.k_a[entering_a]
        departing_k = self.tableau.determine_departing(entering_k)
        if departing_k is None:
            departing_a = None
        else:
            departing_a = self.a_k[departing_k]
        return (departing_a)
    
    def iterate(self,  draw = False, verbose=0):
        entering_as = self.determine_entering_arcs()
        print('entering = ',entering_as)
        if not entering_as:
            if verbose>0:
                print('Optimal solution found.\n=======================')
            if draw:
                mu_a,p_z,g_a = self.musol_a(),self.p0sol_z(),self.gain_a()
                self.draw(p_z = p_z,mu_a=mu_a)
            return(0)
        else:
            entering_a=entering_as[0]
            departing_a = self.determine_departing_arc(entering_a)
            print('entering_a=', entering_a,'departing_a=', departing_a)
            if departing_a is None:
                if verbose>0:
                    print('Unbounded solution.')
                return(1)
            else:
                if verbose>1:
                    print('entering=',entering_a)
                    print('departing=',departing_a)
                if draw:
                    mu_a,p_z,g_a = self.musol_a(),self.p0sol_z(),self.gain_a()
                    self.draw(p_z = p_z,mu_a=mu_a, gain_a = g_a, entering_a = entering_a, departing_a = departing_a)
                    
                self.tableau_update(entering_a,departing_a)
                self.tableau
                return(2)


class EQF_problem:
    def __init__(self, nodesList, arcsList, galois_xy, q_z, active_basis=None, zero_node=0, pos=None, seed=777, verbose=0):

        self.zero_node = zero_node
        self.nbz = len(nodesList)
        self.nba = len(arcsList)
        self.nodesList = nodesList
        self.arcsList = arcsList
        self.nodesDict = {node:node_ind for (node_ind,node) in enumerate(self.nodesList)}
        self.arcsDict = {arc:arc_ind for (arc_ind,arc) in enumerate(self.arcsList)}
        if verbose>1:
            print('Number of nodes = '+str(self.nbz)+'; number of arcs = '+str(self.nba)+'.')

        data = np.concatenate([-np.ones(self.nba),np.ones(self.nba)])
        arcsIndices = list(range(self.nba))
        arcsOrigins = [self.nodesDict[o] for o,d in self.arcsList]
        arcsDestinations = [self.nodesDict[d] for o,d in self.arcsList]
        
        znotzero = [i for i in range(self.nbz) if i != zero_node]
        self.q_z = q_z
        self.q0_z = q_z[znotzero]
        
        self.nabla_a_z = np.array(sp.csr_matrix((data, (arcsIndices+arcsIndices, arcsOrigins+arcsDestinations)), shape = (self.nba,self.nbz)).todense())
        self.nabla0_a_z = self.nabla_a_z[:,znotzero]
        
        self.basis = two_phase(self.nabla0_a_z.T,self.q0_z)
        assert len(self.basis) == (self.nbz - 1)
        
        self.digraph = nx.DiGraph()
        self.digraph.add_edges_from(arcsList)
        if pos is None:
            pos = nx.spring_layout(self.digraph, seed=seed)
        self.pos = pos

        #arcsNames = [str(x)+str(y) for (x,y) in self.arcsList]
        self.galois_xy = galois_xy
        self.create_pricing_tree()
        
        

    def create_pricing_tree(self, verbose = False):
        the_graph = nx.DiGraph()
        the_graph.add_edges_from([self.arcsList[a] for a in self.basis])

        self.tree = {}

        def create_anytree(node, parent=None):
            if node not in self.tree:
                self.tree[node] = Node(name=node, parent=parent)
            else:
                self.tree[node].parent = parent
            for child in list(the_graph.neighbors(node)) + list(the_graph.predecessors(node)):
                if parent is None :
                    create_anytree(child, self.tree[node])
                elif child != parent.name:  # Prevent going back to the parent
                    create_anytree(child, self.tree[node])
                    
        create_anytree(self.nodesList[self.zero_node])
        if verbose:
            self.print_pricing_tree()

    def print_pricing_tree(self):
        for pre, fill, node in RenderTree(self.tree[self.nodesList[self.zero_node]]):
            print("%s%s" % (pre, node.name))
    
    def psol_z(self, current_price=0):
        nodename = self.nodesList[self.zero_node]
        self.set_prices_r(nodename,current_price)
        p_z = np.zeros(self.nbz)
        for (z,thename) in enumerate(self.nodesList):
            p_z[z] = self.tree[thename].price
        return(p_z)
        

    def set_prices_r(self,nodename, current_price=0):
        self.tree[nodename].price = current_price
        for child in self.tree[nodename].children:
            self.set_prices_r(child.name,self.galois_xy[(child.name,nodename)](current_price))


    def cut_pricing_tree(self,a_exiting): # returns root of second connected component
        x,y = self.arcsList[a_exiting]
        print (x,y)
        if (self.tree[y].parent == self.tree[x]):
            return y
        elif (self.tree[x].parent == self.tree[y]):
            return x
        else:
            print('Error in pricing tree during cut phase.')
    
    def paste_pricing_tree(self,a_entering,z_oldroot):
        x,y = self.arcsList[a_entering]

        if (self.tree[z_oldroot] in self.tree[y].ancestors):
            z_newroot , z_prec = y,x
        elif (self.tree[z_oldroot] in self.tree[x].ancestors):
            z_newroot , z_prec = x,y
        else:
            print('Error in pricing tree during paste phase.')
        
        z = z_newroot
        while (z_prec != z_oldroot):
            znext = self.tree[z].parent.name
            self.tree[z].parent = self.tree[z_prec]
            z_prec = z
            z = znext
    
    def iterate(self,  draw = False, verbose=0):
        p_z = self.psol_z()
        cost_improvement_a = np.array([self.galois_xy[(x,y)] (p_z[self.nodesDict[y]]) - p_z[self.nodesDict[x]] for (x,y) in self.arcsList])
        entering_as = np.where(cost_improvement_a )[0].tolist() 
        print('entering = ',entering_as)
        if not entering_as:
            if verbose>0:
                print('Optimal solution found.\n=======================')
            if draw:
                mu_a,p_z,g_a = self.musol_a(),self.p0sol_z(),self.gain_a()
                self.draw(p_z = p_z,mu_a=mu_a)
            return(0)
        else:
            entering_a=entering_as[0]
            departing_a = self.determine_departing_arc(entering_a)
            print('entering_a=', entering_a,'departing_a=', departing_a)
            if departing_a is None:
                if verbose>0:
                    print('Unbounded solution.')
                return(1)
            else:
                if verbose>1:
                    print('entering=',entering_a)
                    print('departing=',departing_a)
                if draw:
                    mu_a,p_z,g_a = self.musol_a(),self.p0sol_z(),self.gain_a()
                    self.draw(p_z = p_z,mu_a=mu_a, gain_a = g_a, entering_a = entering_a, departing_a = departing_a)
                    
                z_oldroot = self.cut_pricing_tree(departing_a)
                self.paste_pricing_tree(entering_a,z_oldroot)
                self.basis.remove(departing_a)
                self.basis.append(entering_a)
                return(2)

##################################################################
##################################################################
##################################################################
                
class Bipartite_EQF_problem:
    def __init__(self, n_x, m_y, galois_xy, label_galois_xy=None, verbose=0):
        self.n_x,self.m_y = n_x,m_y
        self.nbx,self.nby = len(n_x),len(m_y)
        self.nbz = self.nbx + self.nby
        self.nba = self.nbx*self.nby
        self.galois_xy = galois_xy
        self.label_galois_xy = label_galois_xy
        self.p_z = np.zeros(self.nbz)
        
        self.digraph = nx.DiGraph()
        self.digraph.add_nodes_from(['x'+str(x) for x in range(self.nbx)], bipartite=0)
        self.digraph.add_nodes_from(['y'+str(y) for y in range(self.nby)], bipartite=1)
        self.digraph.add_edges_from([('x'+str(x),'y'+str(y)) for x in range(self.nbx) for y in range(self.nby)])
        bottom_nodes, top_nodes = nx.bipartite.sets(self.digraph)
        self.pos = {}
        self.pos.update((node, (1, index)) for index, node in enumerate(bottom_nodes))  # Set one side for one set
        self.pos.update((node, (2, index)) for index, node in enumerate(top_nodes))

        self.create_tree()
        self.update_p_z()
   
    def draw(self, draw_prices=False, mu_a=None, plot_galois=False, entering_a=None, departing_a=None, gain_a=None, figsize=(50, 30)):
        nx.draw(self.digraph, self.pos, with_labels=False)

        if entering_a is not None:
            nx.draw_networkx_edges(self.digraph, self.pos, edgelist=[entering_a],
                                   edge_color='green', connectionstyle='arc3,rad=0.3')

        if departing_a is not None:
            nx.draw_networkx_edges(self.digraph, self.pos, edgelist=[departing_a],
                                   style='dotted', edge_color='white')

        q_z = np.concatenate([self.n_x,self.m_y])
        labels = {z: f"{ ('n'*self.nbx + 'm'*self.nby)[k]}={q_z[k]:.2f}"+'\n'+z+'\n' for k,z in enumerate(self.digraph.nodes()) }
        if draw_prices:
            labels = {z: labels[z] + f"p={self.p_z[k]:.2f}" for k,z in enumerate(self.digraph.nodes())}
        nx.draw_networkx_labels(self.digraph, self.pos, labels,font_size=10,verticalalignment = 'center')

        edge_labels = {e: '' for e in self.digraph.edges()}
        if plot_galois:
            for e in self.digraph.edges():
                edge_labels[e] += self.label_galois_xy[e]
        #if mu_a is not None:
        #    for (i,e) in enumerate(self.digraph.edges()):
        #        if i in self.basis():
        #            edge_labels[e] += '\nμ='+f"{mu_a[i]:.0f}"
        if gain_a is not None:
            for (i,e) in enumerate(self.digraph.edges()):
                if gain_a[i]!=self.c_a[i]: 
                    edge_labels[e] += '\ng='+f"{gain_a[i]:.0f}"
        nx.draw_networkx_edge_labels(self.digraph, self.pos, edge_labels=edge_labels, font_color='red', label_pos=.8)

        plt.figure(figsize=figsize)
        plt.show()

    def create_tree(self, display_tree=False):
        x,y=0,0
        res_x,res_y = self.n_x.copy(),self.m_y.copy()
        current_parent = 'x'+str(x)
        current_parent_node = Node(name=current_parent, parent=None)
        root_node = current_parent_node
        self.tree = {current_parent: root_node}
        current_parent_node.price = 0
        current_parent_node.flow = 0
        while (x<self.nbx) & (y<self.nby):
            current_child = ('y'+str(y) if current_parent[0]=='x' else 'x'+str(x))
            current_child_node = Node(name = current_child, parent = current_parent_node)
            current_child_node.price = self.galois_xy[(current_child,current_parent)](current_parent_node.price)
            self.tree[current_child] = current_child_node
            if res_x[x] <= res_y[y]:
                current_child_node.flow = res_x[x] # (-1)**(current_child[0]=='x') *
                res_x[x], res_y[y] = 0, res_y[y]-res_x[x]
                if current_parent[0]=='x':
                    current_parent = 'y'+str(y)
                    current_parent_node = current_child_node
                x = x+1
            else:
                current_child_node.flow = res_y[y] # (-1)**(current_child[0]=='x') *
                res_x[x], res_y[y] = res_x[x]-res_y[y], 0
                if current_parent[0]=='y':
                    current_parent = 'x'+str(x)
                    current_parent_node = current_child_node
                y = y+1
        if display_tree:
            self.display_tree()
        return root_node
    
    def display_tree(self):
        for pre, fill, node in RenderTree(self.tree['x0']):
            print("%s%s%s%s%s%s" % (pre, node.name,', p=', node.price, ', μ=' , node.flow))
    
    def update_p_z(self, current_price=0):
        for z in range(self.nbz):
            self.p_z[z] = self.tree[list(self.digraph.nodes())[z]].price
        return self.p_z

    def cost_improvement_a(self):
        return np.array([ self.galois_xy[(x,y)](self.p_z[self.nbx + int(y[1:])]) - self.p_z[int(x[1:])] for (x,y) in self.digraph.edges() ])

    def determine_entering_arc(self, tol=1e-5, verbose=0):
        cost_improvement_a = self.cost_improvement_a()
        entering_as = [list(self.digraph.edges())[a] for a in range(self.nba) if cost_improvement_a[a] > tol]
        #entering_as = [(x,y) for (x,y) in self.digraph.edges() if self.p_z[int(x[1:]) < self.galois_xy[(x,y)](self.p_z[self.nbx + int(y[1:])])]
        if verbose>0:
            print('Arbitrable arcs:', entering_as)
        if not entering_as:
            return None
        else:
            return entering_as[0]

    def determine_departing_arc(self, entering_a):
        x,y = entering_a
        ancestors_x = [a.name for a in self.tree[x].path]
        ancestors_y = [a.name for a in self.tree[y].path]
        unique_ancestors_x = [node for node in ancestors_x if node not in ancestors_y]
        unique_ancestors_y = [node for node in ancestors_y if node not in ancestors_x]
        lca = ancestors_x[len(ancestors_x)-len(unique_ancestors_x)-1]
        path_x_to_y = unique_ancestors_x[::-1] + [lca] + unique_ancestors_y
        arcs_x_to_y = [(path_x_to_y[i],path_x_to_y[i+1]) for i in range(len(path_x_to_y)-1)]
        flow_x_to_y = [self.tree[n].flow for n in unique_ancestors_x[::-1]] + [self.tree[n].flow for n in unique_ancestors_y]
        departing_mu, departing_a = min(zip(flow_x_to_y[::2], arcs_x_to_y[::2]))
        return departing_a, departing_mu


    def update_tree(self, entering_a, departing_a, departing_mu):
        x,y = departing_a
        if self.tree[x].parent == self.tree[y]:
            z_oldroot = x
        elif self.tree[y].parent == self.tree[x]:
            z_oldroot = y
        else:
            print('Error in pricing tree during cut phase.')
            return

        x,y = entering_a
        #print(z_oldroot)
        if self.tree[z_oldroot] in self.tree[y].path:
            z_newroot, z_prec = y,x
        elif self.tree[z_oldroot] in self.tree[x].path:
            z_newroot, z_prec = x,y
        else:
            print('Error in pricing tree during paste phase.')
            return

        for i,z in enumerate(self.tree[x].path[1:]): # can be part of determine_departing_arc
            z.flow += (-1)**i * departing_mu
        for i,z in enumerate(self.tree[y].path[1:]):
            z.flow += (-1)**(i+1) * departing_mu

        z = z_newroot
        while z_prec != z_oldroot:
            z_next = self.tree[z].parent.name
            self.tree[z].parent = self.tree[z_prec]
            z_prec = z
            z = z_next

        z = z_oldroot
        while z != z_newroot:
            self.tree[z].flow = self.tree[z].parent.flow
            z = self.tree[z].parent.name
        self.tree[z_newroot].flow = departing_mu

        new_price = self.galois_xy[(z_newroot,self.tree[z_newroot].parent.name)](self.tree[z_newroot].parent.price)
        self.set_prices_r(z_newroot, new_price)


    def cut_pricing_tree(self, departing_a): # returns root of second connected component
        x,y = departing_a
        if self.tree[x].parent == self.tree[y]:
            return x
        elif self.tree[y].parent == self.tree[x]:
            return y
        else:
            print('Error in pricing tree during cut phase.')
    
    def paste_pricing_tree(self, entering_a, z_oldroot, departing_mu):
        x,y = entering_a
        if self.tree[z_oldroot] in self.tree[y].ancestors:
            z_newroot, z_prec = y,x
        elif self.tree[z_oldroot] in self.tree[x].ancestors:
            z_newroot, z_prec = x,y
        else:
            print('Error in pricing tree during paste phase.')
            return

        z = z_newroot
        while z_prec != z_oldroot:
            z_next = self.tree[z].parent.name
            self.tree[z].parent = self.tree[z_prec]
            z_prec = z
            z = z_next

        z = z_oldroot
        while z != z_newroot:
            self.tree[z].flow = self.tree[z].parent.flow
            z = self.tree[z].parent.name
        self.tree[z_newroot].flow = departing_mu

    def iterate(self, draw=False, verbose=0):
        cost_improvement_a = self.cost_improvement_a()
        entering_a = self.determine_entering_arc(verbose=verbose-2)
        if entering_a is None:
            if verbose>0:
                print('Optimal solution found.\n=======================')
            return 0
        else:
            departing_a, departing_mu = self.determine_departing_arc(entering_a)
            if verbose>1:
                print(str(entering_a) + ' enters, ' + str(departing_a) + ' departs')
            if departing_a is None:
                if verbose>0:
                    print('Unbounded solution.')
                return 1
            else:
                self.update_tree(entering_a, departing_a, departing_mu)
                self.update_p_z()
                return 2

    def set_prices_r(self, z, current_price=0):
        self.tree[z].price = current_price
        for child in self.tree[z].children:
            self.set_prices_r(child.name, self.galois_xy[(child.name,z)](current_price))

