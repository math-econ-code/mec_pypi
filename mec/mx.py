# Econometrics library

import numpy as np, scipy.sparse as sp, pandas as pd


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




    