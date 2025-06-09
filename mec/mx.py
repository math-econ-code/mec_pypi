# Econometrics library

import numpy as np, scipy.sparse as sp, pandas as pd


def ivgmm(Y_i,X_i_k,Z_i_l,W_l_l=None, require_der = 0):
    I = len(Y_i)
    if W_l_l is None:
        W_l_l = np.linalg.inv( Z_i_l.T @ Z_i_l /  I )
    Pi_i_i =  Z_i_l @ W_l_l @ Z_i_l.T
    beta_k = np.linalg.solve(X_i_k.T @ Pi_i_i @ X_i_k,  X_i_k.T @ Pi_i_i @ Y_i)
    val = (Y_i - X_i_k @ beta_k).T @ Pi_i_i @ (Y_i - X_i_k @ beta_k) / (2* I * I)
    if require_der > 0:
        dval_dY_i = (Pi_i_i @ (Y_i - X_i_k @ beta_k) / (I*I)).flatten()
        return beta_k,val,dval_dY_i
    else:
        return beta_k,val

def efficient_ivgmm(Y_i,X_i_k,Z_i_l, centering = True):
    I=len(Y_i)
    beta1_k = ivgmm(Y_i,X_i_k,Z_i_l )[0] # first stage obtained by 2SLS
    epsilon1_i = Y_i - X_i_k @ beta1_k
    mhat_l_i = Z_i_l.T * epsilon1_i[None,:]
    mbar_l = mhat_l_i.mean(axis=1)
    Sigmahat1_l_l = (mhat_l_i @ mhat_l_i.T) / I - centering * mbar_l[:,None] * mbar_l[None,:]
    W_l_l =  np.linalg.inv(Sigmahat1_l_l)
    beta2_k, val2 = ivgmm(Y_i,X_i_k,Z_i_l,W_l_l )
    return beta2_k,val2
