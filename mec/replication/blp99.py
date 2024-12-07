# This code draws upon:
# * Berry Levinsohn Pakes (1999) "Voluntary Export Restraints on Automobiles: Evaluating a Strategic Trade Policy" (https://www.jstor.org/stable/2171802)
# * Gentzkow and Shapiro (2015). "Code and data for the analysis of the Berry Levinsohn Pakes method of moments paper" (https://scholar.harvard.edu/files/shapiro/files/blp_replication.pdf)
# * Conlon and Gortmaker (2020). Python code for BLP estimation (https://github.com/jeffgortmaker/pyblp)
# * Rainie Lin, 2021. Python code for BLP estimation (https://github.com/ranielin/Berry-Levinsohn-and-Pakes-1995-Replication)

import numpy as np, pandas as pd
import mec.blp
from mec.data import load_blp_car_data
from mec.blp import create_blp_instruments, pi_inv

def construct_car_variables_from_blp():
    prod,agents = load_blp_car_data()
    mkt_o = prod['market_ids'].to_numpy()
    #
    Xs_y_ind = mec.blp.organize_markets(mkt_o, np.block([np.ones( (obs,1) ), prod[ ['hpwt','air','mpd','space' ]] ]))
    #
    firms_y = mec.blp.organize_markets(mkt_o,prod['firm_ids'].to_numpy())
    ps_y = mec.blp.organize_markets(mkt_o,prod['prices'].to_numpy())
    pis_y = mec.blp.organize_markets(mkt_o,prod['shares'].to_numpy())
    cars_y = mec.blp.organize_markets(mkt_o,prod['car_ids'].to_numpy())
    #
    Zs_y_ind = mec.blp.organize_markets(mkt_o, create_blp_instruments(mec.blp.collapse_markets(mkt_o,Xs_y_ind), prod[['market_ids','firm_ids','car_ids']] ))
    #
    theW = np.block([np.ones( (obs,1) ), prod[ ['hpwt','air','mpg','space','trend' ]] ])
    theW[:,[1,3,4] ]= np.log(theW[:,[1,3,4] ])
    theZS = create_blp_instruments(theW , prod[['market_ids','firm_ids','car_ids']] )
    theZS[:,-1] = prod['mpd'].to_numpy()
    theZS[:,5] += 71.
    ZSs_y_ind = mec.blp.organize_markets(mkt_o,theZS)
    xis_y_l = [ np.concatenate([p_y.reshape((-1,1)),X_y_ind],axis=1) for (p_y,X_y_ind) in zip(ps_y,Xs_y_ind) ]
    return mkt_o,Xs_y_ind,firms_y,ps_y,pis_y,Zs_y_ind,ZSs_y_ind ,xis_y_l