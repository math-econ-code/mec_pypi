import numpy as np
import pkg_resources
import pandas as pd
import tabulate as tb

def load_stigler_data(nbi = 9, nbj = 77, verbose=False):

    thepath =data_file_path = pkg_resources.resource_filename('mec', 'datasets/stigler-diet/StiglerData1939.txt')
    thedata = pd.read_csv(thepath , sep='\t')
    thedata = thedata.dropna(how = 'all')
    commodities = (thedata['Commodity'].values)[:-1]
    allowance = thedata.iloc[-1, 4:].fillna(0).transpose()
    nbi = min(len(allowance),nbi)
    nbj = min(len(commodities),nbj)
    if verbose:
        print('Daily nutrient content:')
        print(tb.tabulate(thedata.head()))
        print('\nDaily nutrient requirement:')
        print(allowance)
    return({'N_i_j':thedata.iloc[:nbj, 4:(4+nbi)].fillna(0).to_numpy().T,
            'd_i':np.array(allowance)[0:nbi],
            'c_j':np.ones(len(commodities))[0:nbj],
            'names_i': list(thedata.columns)[4:(4+nbi)],
            'names_j':commodities[0:nbj]}) 


def load_DupuyGalichon_data( verbose=False):
    thepath =data_file_path = pkg_resources.resource_filename('mec', 'datasets/marriage_personality-traits/')
    data_X = pd.read_csv(thepath + "Xvals.csv")
    data_Y = pd.read_csv(thepath + "Yvals.csv")
    aff_data = pd.read_csv(thepath + "affinitymatrix.csv")
    A_k_l = aff_data.iloc[0:nbk,1:nbl+1].values

    if verbose:
        print(data_X.head())
        print(data_Y.head())
        print(tb.tabulate(A_k_l))
        
    return({'data_X': data_X,
            'data_Y': data_Y,
            'A_k_l': A_k_l})