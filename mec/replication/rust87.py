# This code draws upon:
# * Rust (1987) "Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher" (https://www.jstor.org/stable/1911257)
# Ranie Lin, Rust (1987) replication. Python project, available on github https://github.com/ranielin/Rust-1987-Replication.


import numpy as np, pandas as pd, scipy.sparse as sp
from mec.data import load_Rust_data

def build_rust_model_primitives(X=90):
    thedata = load_Rust_data()
    theta30 = (thedata[:,2]==0).mean()
    theta31 = (thedata[:,2]==1).mean()
    theta32 = 1-theta30 - theta31
    P_xp_x = theta30 * np.eye(X) + np.diag([theta31]*(X-1),-1) + np.diag([theta32]*(X-2),-2)
    P_xp_x[X-1,X-2] += theta32
    P_xp_x[X-1,X-1] = 1
    P_xp_x_y = np.zeros((X,X,2))
    P_xp_x_y[:,:,0] = P_xp_x
    P_xp_x_y[:,:,1] = P_xp_x[:,0][:,None] 
    #
    phi_x_y_k = np.zeros((X,2,2))
    phi_x_y_k[:,1,0] = -1.0 # if repair, replacement cost is lambda_0
    phi_x_y_k[:,1,1] = 0.0 # if repair, replacement cost does not depend on mileage 
    phi_x_y_k[:,0,0] = 0.0 # if no repair, operating cost independent of the repair cost
    phi_x_y_k[:,0,1] = -0.001 * np.arange(X,dtype = np.float64) # if no repair, operating cost proportional to mileage   
    muhat_x_y = np.zeros((X,2))
    for x in range(X):
        muhat_x_y[x,0] = ((thedata[:,0] ==x) & (thedata[:,1]==0)).sum()
        muhat_x_y[x,1] = ((thedata[:,0] ==x) & (thedata[:,1]==1)).sum()
    return sp.csr_matrix(P_xp_x_y.reshape((X,-1))), phi_x_y_k, muhat_x_y.flatten()
