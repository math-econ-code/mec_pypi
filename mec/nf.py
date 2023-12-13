# Network flow problems library
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from mec.lp import Tableau, two_phase

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


class Network_problem:
    def __init__(self,nodesList,arcsList,c_a,q_z , active_basis = None, zero_node = 0, pos=None,seed=777, verbose=0):
        self.zero_node = zero_node
        self.nbz = len(nodesList)
        self.nba = len(arcsList)
        self.nodesList = nodesList
        self.arcsList = arcsList
        self.c_a = c_a
        self.nodesDict = {node:node_ind for (node_ind,node) in enumerate(self.nodesList)}
        self.arcsDict = {arc:arc_ind for (arc_ind,arc) in enumerate(self.arcsList)}
        if verbose>1:
            print('Number of nodes='+str(self.nbz)+'; number of arcs='+str(self.nba)+'.')

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
        assert len(active_basis)== (self.nbz - 1)
        
        self.digraph = nx.DiGraph()
        self.digraph.add_edges_from(arcsList)
        if pos is None:
            pos = nx.spring_layout(self.digraph, seed=seed)
        self.pos = pos

        
        arcsnames = [str(x)+str(y) for (x,y) in self.arcsList]

        self.tableau = Tableau(A_i_j = np.asarray(np.linalg.solve(self.B(active_basis),self.N(active_basis))),
                                     d_i = self.q0_z,
                                     c_j = self.gain_a(active_basis)[self.nonbasis(active_basis)],
                                     slack_var_names_i = [arcsnames[i] for i in active_basis] ,
                                     decision_var_names_j = [arcsnames[i] for i in self.nonbasis(active_basis)]
                                    )
        the_arcs_indices = list(active_basis) + list(set(range(self.nba)) - set(active_basis))
        self.a_k = {k: the_arcs_indices[k] for k in range(self.nba)}
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
                with_labels=True)
                                              
        nx.draw_networkx_edges(self.digraph, self.pos, edgelist=[self.arcsList[i] for i in self.basis()], 
                               edge_color='blue',) #connectionstyle='arc3,rad=0.3'
        
        labels = {z: f"q={self.q_z[i]:.0f}" for i,z in enumerate(self.nodesList) }
        label_pos = {z: (position[0], position[1] - 0.03) for z, position in self.pos.items()}        
        if p_z is not None:
            p_z = np.concatenate([np.zeros(1),p_z])
            labels = {z: labels[z]+ f"\np={p_z[i]:.0f}" for i,z in enumerate(self.nodesList)} 
        
        nx.draw_networkx_labels(self.digraph, label_pos, labels,font_size=8)
            
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
        
    def psol_z(self,basis = None):
        return (np.linalg.solve(self.B(basis).T,self.c_a[self.basis(basis)]))
    
    def gain_a(self, basis = None):
        p_z = self.psol_z(basis)
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
        if not entering_as:
            if verbose>0:
                print('Optimal solution found.\n=======================')
            if draw:
                mu_a,p_z,g_a = self.musol_a(),self.psol_z(),self.gain_a()
                self.draw(p_z = p_z,mu_a=mu_a)
            return(0)
        else:
            entering_a=entering_as[0]
            departing_a = self.determine_departing_arc(entering_a)
            if departing_a is None:
                if verbose>0:
                    print('Unbounded solution.')
                return(1)
            else:
                if verbose>1:
                    print('entering=',entering_a)
                    print('departing=',departing_a)
                if draw:
                    mu_a,p_z,g_a = self.musol_a(),self.psol_z(),self.gain_a()
                    self.draw(p_z = p_z,mu_a=mu_a, gain_a = g_a, entering_a = entering_a, departing_a = departing_a)
                    
                self.tableau_update(entering_a,departing_a)
                return(2)
                

