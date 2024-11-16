# Econometrics library

import numpy as np
import scipy.sparse as sp


def iv_gmm(Y_i,X_i_k,Z_i_l, efficient=False, centering = True):
    def beta_gmm(Y_i,X_i_k,Z_i_l,W_l_l ):
        ZtildeT_k_i = X_i_k.T @ Z_i_l @ W_l_l @ Z_i_l.T
        return np.linalg.solve(ZtildeT_k_i @ X_i_k,ZtildeT_k_i @ Y_i)
    I=len(Y_i)
    W_l_l = np.linalg.inv( Z_i_l.T @ Z_i_l / I)
    beta_k = beta_gmm(Y_i,X_i_k,Z_i_l,W_l_l ) # first stage obtained by 2SLS
    if efficient:
        epsilon_i = Y_i - X_i_k @ beta_k
        mhat_l_i = Z_i_l.T * epsilon_i[None,:]
        mbar_l = mhat_l_i.mean(axis=1)
        Sigmahat_l_l = (mhat_l_i @ mhat_l_i.T) / I - centering * mbar_l[:,None] * mbar_l[None,:]
        W_l_l =  np.linalg.inv(Sigmahat_l_l)
        beta_k = beta_gmm(Y_i,X_i_k,Z_i_l,W_l_l )
    Pi_i_i = Z_i_l @ W_l_l @ Z_i_l.T
    XPiY_k_i = X_i_k.T @ Pi_i_i @ Y_i
    objval = (Y_i.T @ Pi_i_i @ Y_i - XPiY_k_i.T @ np.linalg.inv(  X_i_k.T @ Pi_i_i @ X_i_k ) @ XPiY_k_i )/ (2*I*I)
    return beta_k,objval


def partGstar(pi_t_y,theLambda_k_k,epsilon_t_i_k, xi_k_y,maxit = 100000, reltol=1E-8):
    (K ,Y ) = xi_k_y.shape
    (T,I,_) = epsilon_t_i_k.shape
    n_t_i = np.ones((T,1)) @ np.ones((1,I)) / I
    varepsilon_t_i_y = (epsilon_t_i_k.reshape((-1,K)) @ theLambda_k_k @ xi_k_y).reshape((T,I,Y))
    U_t_y = np.zeros((T,Y))
    for i in range(maxit): # ipfp
        max_t_i = np.maximum((U_t_y[:,None,:] + varepsilon_t_i_y).max(axis = 2),0)
        u_t_i = max_t_i + np.log ( np.exp(- max_t_i ) + np.exp(U_t_y[:,None,:] + varepsilon_t_i_y - max_t_i[:,:,None]).sum(axis = 2) ) - np.log(n_t_i)
        max_t_y = (varepsilon_t_i_y - u_t_i[:,:,None]).max(axis=1)
        Up_t_y = - max_t_y -np.log( np.exp(varepsilon_t_i_y - u_t_i[:,:,None] - max_t_y[:,None,:] ).sum(axis=1)  / pi_t_y)
        if (np.abs(Up_t_y-U_t_y) < reltol * (np.abs(Up_t_y)+np.abs(U_t_y))/2).all():
            break
        else:
            U_t_y = Up_t_y

    pi_t_i_y = np.concatenate( [ np.exp(U_t_y[:,None,:]+ varepsilon_t_i_y - u_t_i[:,:,None] ), 
                                np.exp( - u_t_i)[:,:,None]],axis=2)
    
    Sigma = sp.kron(sp.eye(T),sp.bmat([[sp.kron( sp.eye(I),      np.ones((1,Y+1)))            ],
                                   [sp.kron(np.ones((1,I)),  sp.diags([1],shape=(Y,Y+1)))]]) )
    Deltapi = sp.diags(pi_t_i_y.flatten())
    proj = sp.kron(sp.eye(T),sp.kron( sp.eye(I), sp.diags([1],shape=(Y+1,Y)).toarray()) )
    A = (Sigma @ Deltapi @ Sigma.T).tocsc()
    B = (Sigma @ Deltapi @ proj @ sp.kron( epsilon_t_i_k.reshape((-1,K)) , xi_k_y.T )).tocsc()
    dUdLambda_t_y_k_k = - sp.linalg.spsolve(A,B).toarray().reshape((T,I+Y,K,K))[:,-Y:,:,:]
    return(U_t_y, dUdLambda_t_y_k_k)