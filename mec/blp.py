import numpy as np, scipy.sparse as sp, pandas as pd


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
 
def build_nus(eta_t_i_l, xis_y_k):
    I, L, K = eta_t_i_l.shape[1], eta_t_i_l.shape[2],xis_y_k[0].shape[1]
    M = L * K
    nus_i_y_m = []
    for t,xi_y_k in enumerate(xis_y_k):
        nu_i_y_l_k = eta_t_i_l[t,:,None,:,None] * xi_y_k[None,:,None,:] 
        nus_i_y_m.append(nu_i_y_l_k.reshape((I,-1,M)))
    return nus_i_y_m

def build_dnudps(eta_t_i_l, xis_y_k, index_price = 0):
    (T,I,L), K =  eta_t_i_l.shape , xis_y_k[0].shape[1]
    M = L * K
    thevec = np.zeros(K)
    thevec[index_price] = 1
    dnudps_i_y_m = []
    for t in range(T):
        dnudp_i_y_l_k = eta_t_i_l[t,:,None,:,None] * thevec[None,None,None,:] 
        dnudps_i_y_m.append(dnudp_i_y_l_k.reshape((I,-1,M)))
    return dnudps_i_y_m
 
def pi_inv_t(pi_t_y,nu_t_i_y_m, tau_m, maxit = 100000, reltol=1E-8, require_der = 0):
    (T,I,Y,M) = nu_t_i_y_m.shape
    n_t_i = np.ones((T,I)) / I
    nutau_t_i_y = (nu_t_i_y_m.reshape((-1,M)) @ tau_m ).reshape((T,I,Y))
    U_t_y = np.zeros((T,Y))
    for i in range(maxit): # ipfp
        max_t_i = np.maximum((U_t_y[:,None,:] + nutau_t_i_y).max(axis = 2),0)
        u_t_i = max_t_i + np.log ( np.exp(- max_t_i ) + np.exp(U_t_y[:,None,:] + nutau_t_i_y - max_t_i[:,:,None]).sum(axis = 2) ) - np.log(n_t_i)
        max_t_y = (nutau_t_i_y - u_t_i[:,:,None]).max(axis=1)
        Up_t_y = - max_t_y -np.log( np.exp(nutau_t_i_y - u_t_i[:,:,None] - max_t_y[:,None,:] ).sum(axis=1)  / pi_t_y)
        if (np.abs(Up_t_y-U_t_y) < reltol * (np.abs(Up_t_y)+np.abs(U_t_y))/2).all():
            break
        else:
            U_t_y = Up_t_y

    res = [U_t_y]
    if require_der>0:
        pi_t_i_y = np.concatenate( [ np.exp(U_t_y[:,None,:]+ nutau_t_i_y - u_t_i[:,:,None] ), 
                                    np.exp( - u_t_i)[:,:,None]],axis=2)
        
        Sigma = sp.kron(sp.eye(T),sp.bmat([[sp.kron( sp.eye(I),      np.ones((1,Y+1)))    ],
                                    [sp.kron(np.ones((1,I)),  sp.diags([1],shape=(Y,Y+1)))]]) )
        Deltapi = sp.diags(pi_t_i_y.flatten())
        proj = sp.kron(sp.eye(T),sp.kron( sp.eye(I), sp.diags([1],shape=(Y+1,Y)).toarray()) )
        A = (Sigma @ Deltapi @ Sigma.T).tocsc()
        B = (Sigma @ Deltapi @ proj @ nu_t_i_y_m.reshape((-1,M)) ) 
        dUdtau_t_y_m = - sp.linalg.spsolve(A,B).reshape((T,I+Y,M))[:,-Y:,:]
        res.append(dUdtau_t_y_m)

    return res


def pi_inv(pi_y,nu_i_y_m, tau_m, maxit = 100000, reltol=1E-8, require_der = 0 ):
    res = pi_inv_t(pi_y[None,:],nu_i_y_m[None,:,:,:] ,tau_m,  maxit , reltol, require_der )
    return [res[i].squeeze(axis=0) for i in range(require_der+1)]
    


def organize_markets(markets_o, series_o):
    flatten =  (len(series_o.shape)==1) or (series_o.shape[1] ==1)
    vs_y =[]
    for mkt in sorted(set(markets_o)):
        observations = np.where(markets_o == mkt)[0]
        if flatten:
            vs_y.append(series_o.flatten()[observations])
        else:
            vs_y.append(series_o[observations,:])
    return vs_y


def collapse_markets(markets_o,vs_y):
    O = len(markets_o)
    if (len(vs_y[0].shape)==1):
        dimv = 1
    else:
        dimv = vs_y[0].shape[1]
    series_o = np.zeros((O,dimv))
    for mkt,v_y in zip(sorted(set(markets_o)),vs_y):
        observations = np.where(markets_o == mkt)[0]
        series_o[observations,:] = v_y.reshape((-1,dimv))
    return series_o.flatten() if (dimv == 1) else series_o


def compute_shares(Us_y,nus_i_y_m, tau_m):
    pis_y = []
    for (U_y,nu_i_y_m) in zip(Us_y,nus_i_y_m):
        nutau_i_y = nu_i_y_m @ tau_m
        pi_y = (np.exp(U_y[None,:] + nutau_i_y ) / (1+ np.exp( U_y[None,:] + nutau_i_y ).sum(axis= 1) )[:,None] ).mean(axis=0)
        pis_y.append(pi_y)
    return pis_y


def compute_utilities(pis_y,nus_i_y_m, tau_m, require_der = 0 ):
    Us_y = []
    if require_der>0:
        dUs_y_m = []
    for (pi_y,nu_i_y_m) in zip(pis_y,nus_i_y_m):
        res_inversion = pi_inv(pi_y,nu_i_y_m, tau_m, require_der = require_der )
        U_y = res_inversion[0].flatten()
        Us_y.append(U_y)
        if require_der>0:
            dU_y_m = res_inversion[1]
            dUs_y_m.append(dU_y_m)
    
    return [Us_y] if require_der == 0 else [Us_y,dUs_y_m]


def compute_Ds(Us_y,nus_i_y_m,dnudp_i_y_m, tau_m,firms_y, require_der = 0):
    Ds_y_y = []
    for (U_y,nu_i_y_m,dnudp_i_y_m,firm_y) in zip(Us_y,nus_i_y_m, dnudp_i_y_m, firms_y):
        Y = len(U_y)
        H_y_y = 1*(firms_y[:,None] == firms_y[None,:])


        nutau_i_y = nu_i_y_m @ tau_m
        dnutaudp_i_y = dnudp_i_y_m @ tau_m
        pi_i_y = (np.exp(U_y[None,:] + nutau_i_y ) / (1+ np.exp( U_y[None,:] + nutau_i_y ).sum(axis= 1) )[:,None] )
        I_y_y = np.eye(Y) 
        dpidu_i_y_y =   pi_i_y[:,:,None] *( I_y_y[None,:,:] - pi_i_y[:,None,:] )
        dpidp_y_y = ( dnutaudp_i_y[:,None,:] * dpidu_i_y_y ).mean(axis= 0)
        # if require_der>0:
        #     term0 = I_y_y[None,:,:,None] * I_y_y[None,:,None,:]
        #     term1 = - I_y_y[None,:,:,None] * pi_i_y[:,None,None,:] 
        #     term2 = - I_y_y[None,:,None,:] * pi_i_y[:,None,:,None] 
        #     term3 = - I_y_y[None,None,:,:] * pi_i_y[:,None,:,None]
        #     term4 =  2 * pi_i_y[:,None,None,:]*  pi_i_y[:,None,:,None]
        #     d2pidu_i_y_y_y =    pi_i_y[:,:,None,None] *(term0  + term1 + term2 + term3 + term4  )
        #     d2pidp_y_y_y = ( dnutaudp_i_y[:,None,:,None] * dnutaudp_i_y[:,None,None,:] * d2pidu_i_y_y_y).mean(axis= 0)


        Ds_y_y.append(- dpidp_y_y)
        # if require_der>0:
        #     Ds_y_y.append(d
    return Ds_y_y

def compute_D(Us_y,nus_i_y_m,dnudp_i_y_m, tau_m,firms_y ):
    return sp.block_diag(compute_Ds(Us_y,nus_i_y_m, dnudp_i_y_m, tau_m,firms_y ) )


def compute_marginal_costs( Us_y,ps_y,pis_y,nus_i_y_m, dnudp_i_y_m, tau_m,firms_y ):
    mcs_y = []
    Ds_y_y = compute_Ds(Us_y,nus_i_y_m, dnudp_i_y_m, tau_m,firms_y)
    for (p_y,pi_y,D_y_y) in zip(ps_y,pis_y,Ds_y_y): 
        mc_y = p_y - np.linalg.solve(D_y_y,pi_y) # first-order Bertrand equilibrium foc
        mc_y[mc_y < 0] = 0.001 # marginal costs must be nonnegative
        mcs_y.append(mc_y)
    return mcs_y
