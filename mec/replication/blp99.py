# This code draws upon:
# * Berry Levinsohn Pakes (1999) "Voluntary Export Restraints on Automobiles: Evaluating a Strategic Trade Policy" (https://www.jstor.org/stable/2171802)
# * Gentzkow and Shapiro (2015). "Code and data for the analysis of the Berry Levinsohn Pakes method of moments paper" (https://scholar.harvard.edu/files/shapiro/files/blp_replication.pdf)
# * Conlon and Gortmaker (2020). Python code for BLP estimation (https://github.com/jeffgortmaker/pyblp)
# * Rainie Lin (2021). Python code for BLP estimation (https://github.com/ranielin/Berry-Levinsohn-and-Pakes-1995-Replication)

import numpy as np, pandas as pd
from mec.data import load_blp_car_data
from mec.blp import create_blp_instruments, organize_markets, collapse_markets,build_epsilons,build_depsilonsdp


def eta_from_lin(T=20, K=5, nu_mean=0 , nu_var=1, D_mean = None, D_var=None, I=500, seed=123 ): # I = number of simulations per market
    # borrows code from rainie lin for consistency
    if D_mean is None:
            D_mean = np.array([2.01156, 2.06526, 2.07843, 2.05775, 2.02915, 2.05346, 2.06745,
                               2.09805, 2.10404, 2.07208, 2.06019, 2.06561, 2.07672, 2.10437,
                               2.12608, 2.16426, 2.18071, 2.18856, 2.2125 , 2.18377])
    if D_var is None:
         D_var = 2.9584*np.ones(20)
    np.random.seed(seed)
    nu_alpha = np.zeros((1, I, T))
    nu_beta = np.random.normal(nu_mean, pow(nu_var, 0.5), (K, I, T))
    nu = np.concatenate((nu_alpha, nu_beta), axis = 0)
    log_y = np.transpose(np.random.normal(D_mean, pow(D_var, 0.5), (I, 1, T)), (1, 0, 2))
    D = 1/np.exp(log_y)
    eta_t_i_ind = np.concatenate([nu,-D],axis = 0).transpose((2,1,0)) 
    return eta_t_i_ind

def tau_from_lin(theta_2 = np.array([43.501, 3.612, 4.628, 1.818, 1.05, 2.056]),K=5 ):
    # for consistency with rainie lin's code
    gamma = np.zeros((K + 1, 1))
    gamma[0] = theta_2[0]
    sigma = np.zeros((K + 1, K + 1))
    np.fill_diagonal(sigma, np.append([0], theta_2[1:(K + 1)]))
    return np.block([[sigma.T],[gamma.flatten()]]).flatten()



def construct_car_blp_model(lin_compatible = False):
    prod,_ = load_blp_car_data()
    prod['ones'] = 1.
    mkt_o = prod['market_ids'].to_numpy()
    firms_y = organize_markets(mkt_o,prod['firm_ids'].to_numpy()) # firms of each product
    ps_y = organize_markets(mkt_o,prod['prices'].to_numpy()) # prices of each product
    pis_y = organize_markets(mkt_o,prod['shares'].to_numpy()) # shares of each product
    O = len(mkt_o) # number of observations
    # phis_y_k are the regression matrices for demand side (not interacting with individual characteristics)
    phis_y_k = organize_markets(mkt_o, prod[['ones', 'hpwt','air','mpd','space' ]].to_numpy() )
    # xis_y_ind are the regression matrices for the demand side (interacting with indivudual characteristics)
    xis_y_ind = organize_markets(mkt_o, prod[['prices','ones', 'hpwt','air','mpd','space' ]].to_numpy() )
    # zetas_y_d are the instruments for demand side
    zetas_y_d = organize_markets(mkt_o, create_blp_instruments(collapse_markets(mkt_o,phis_y_k), prod[['market_ids','firm_ids','car_ids']] ))
    # gammas_y_l are the regression matrices for supply side, and chis_y_s are the instruments for supply side
    thegamma =  prod[['ones','hpwt','air','mpg','space','trend' ]].to_numpy()
    thegamma[:,[1,3,4] ]= np.log(thegamma[:,[1,3,4] ])
    if lin_compatible:
        thechi = create_blp_instruments(thegamma , prod[['market_ids','firm_ids','car_ids']] )
        thegamma[:,-1] += 71.
    else:
        thechi = create_blp_instruments(thegamma , prod[['market_ids','firm_ids','car_ids']] )
    gammas_y_l = organize_markets(mkt_o,thegamma)
    thechi[:,-1] = prod['mpd'].to_numpy()
    if lin_compatible:
         thechi[:,5] += 71.
    chis_y_s = organize_markets(mkt_o,thechi)
    #
    # eta is the vector of unobservable agents characteristics
    eta_t_i_ind = eta_from_lin( )
    #
    epsilons_i_y_m = build_epsilons(eta_t_i_ind, xis_y_ind)
    depsilonsdp_i_y_m = build_depsilonsdp(eta_t_i_ind, xis_y_ind)
    return mkt_o,phis_y_k,gammas_y_l, firms_y,ps_y,pis_y,zetas_y_d,chis_y_s ,epsilons_i_y_m,depsilonsdp_i_y_m