import numpy as np, scipy.sparse as sp



def create_blp_instruments(X, mkts_firms_prods,include_ones = False, include_arguments = True ):
    if include_ones:
        X = np.block([[np.ones((X.shape[0],1)), X ]] )
    df = pd.DataFrame()
    names = [str(i) for i in range(X.shape[1])]
    df[ names ]=X
    df[['mkt','firm','prod']] = mkts_firms_prods
    thelist1, thelist2 = [], []
    for _, theserie in df[ names ].items():
        thelist1.append ([theserie[(df['mkt']==df['mkt'][i]) & 
                                (df['firm']==df['firm'][i]) & 
                                (df['prod']!=df['prod'][i])  ].sum() for i,_ in df.iterrows() ])

        thelist2.append([theserie[(df['mkt']==df['mkt'][i]) & 
                                (df['firm']!=df['firm'][i]) ].sum() for i,_ in df.iterrows() ])
    if include_arguments:
        return np.block([[X,np.array(thelist1+thelist2).T]])
    else:
        return np.array(thelist1+thelist2).T
 


def organize_markets(markets_o, vec_o):
    flatten =  (len(vec_o.shape)==1) or (vec_o.shape[1] ==1)
    vs_y =[]
    for mkt in sorted(set(markets_o)):
        observations = np.where(markets_o == mkt)[0]
        if flatten:
            vs_y.append(vec_o.flatten()[observations])
        else:
            vs_y.append(vec_o[observations,:])
    return vs_y

def collapse_markets(markets_o,vs_y):
    O = len(markets_o)
    if (len(vs_y[0].shape)==1):
        dimv = 1
    else:
        dimv = vs_y[0].shape[1]
    vec_o = np.zeros((O,dimv))
    for mkt,v_y in zip(sorted(set(markets_o)),vs_y):
        observations = np.where(markets_o == mkt)[0]
        vec_o[observations,:] = v_y.reshape((-1,dimv))
    return vec_o.flatten() if (dimv == 1) else vec_o

def compute_shares(Us_y,epsilon_t_i_k, xis_y_l,thelambda_k_l):
    pis_y = []
    for (t,(U_y,xi_y_l)) in enumerate(zip(Us_y,xis_y_l)):
        epsilon_i_k = epsilon_t_i_k[t,:,:]
        varepsilon_i_y = epsilon_i_k @ thelambda_k_l @ xi_y_l.T
        pi_y = (np.exp(U_y[None,:] + varepsilon_i_y ) / (1+ np.exp( U_y[None,:] + varepsilon_i_y ).sum(axis= 1) )[:,None] ).mean(axis=0)
        pis_y.append(pi_y)
    return pis_y

def compute_utilities(pis_y,epsilon_t_i_k, xis_y_l,thelambda_k_l):
    Us_y = []
    (K,L)=thelambda_k_l.shape
    for (t,(pi_y,xi_y_l)) in enumerate(zip(pis_y,xis_y_l)):
        epsilon_i_k = epsilon_t_i_k[t,:,:]        
        U_y = pi_inv(pi_y,thelambda_k_l,epsilon_i_k, xi_y_l.T)[0].flatten()
        Us_y.append(U_y)
    return Us_y

def compute_omegas(Us_y,epsilon_t_i_k, xis_y_l,thelambda_k_l,firms_y ):
    omegas_y_y = []
    for (t,(U_y,xi_y_l,firm_y)) in enumerate(zip(Us_y,xis_y_l, firms_y)):
        Y = len(U_y)
        epsilon_i_k = epsilon_t_i_k[t,:,:]
        varepsilon_i_y = epsilon_i_k @ thelambda_k_l @ xi_y_l.T
        epslambda0_i = epsilon_i_k @ thelambda_k_l[:,0]
        pi_i_y = (np.exp(U_y[None,:] + varepsilon_i_y ) / (1+ np.exp( U_y[None,:] + varepsilon_i_y ).sum(axis= 1) )[:,None] )
        jacobian_i_y_y =  - epslambda0_i[:,None,None] * (pi_i_y[:,:,None] * np.eye(Y)[None,:,:] - pi_i_y[:,:,None] * pi_i_y[:,None,:] )
        deriv_shares_y_y =jacobian_i_y_y.mean(axis= 0)
        for y in range(Y):
            for yprime in range(y+1):
                if (firm_y[y]!=firm_y[yprime]):
                    deriv_shares_y_y[y,yprime] = 0
                    deriv_shares_y_y[yprime,y] = 0
        omegas_y_y.append(deriv_shares_y_y)
    return omegas_y_y

def compute_omega(Us_y,epsilon_t_i_k, xis_y_l,thelambda_k_l,firms_y ):
    return sp.block_diag(compute_omegas(Us_y,epsilon_t_i_k, xis_y_l,thelambda_k_l,firms_y ) )

def compute_inv_omega(markets_o, U_o,epsilon_t_i_k, xi_o_l,thelambda_k_l,firms_o):
    deriv_shares = compute_omegas(Us_y,epsilon_t_i_k, xis_y_l,thelambda_k_l,firms_y )
    return sp.block_diag( [np.linalg.inv(block) for block in deriv_shares] )

def compute_marginal_costs( Us_y,ps_y,pis_y,epsilon_t_i_k, xis_y_l,thelambda_k_l,firms_y ):
    mcs_y = []
    omegas_y_y = compute_omegas(Us_y,epsilon_t_i_k, xis_y_l,thelambda_k_l,firms_y)
    for (t,(U_y,p_y,pi_y,xi_y_l,firm_y,omega_y_y)) in enumerate(zip(Us_y,ps_y,pis_y,xis_y_l, firms_y,omegas_y_y)): 
        mc_y = p_y - np.linalg.solve(omega_y_y,pi_y) # first-order Bertrand equilibrium foc
        mc_y[mc_y < 0] = 0.001 # marginal costs must be nonnegative
        mcs_y.append(mc_y)
    return mcs_y
