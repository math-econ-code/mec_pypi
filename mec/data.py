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


def print_optimal_diet_stigler_data(q_j):
    print('***Optimal solution***')
    total,thelist = 0.0, []
    for j, commodity in enumerate(commodities):
        if q_j[j] > 0:
            total += q_j[j] * 365
            thelist.append([commodity,q_j[j]])
    thelist.append(['Total cost (optimal):', total])
    print(tb.tabulate(thelist))

def load_DupuyGalichon_data( verbose=False):
    thepath =data_file_path = pkg_resources.resource_filename('mec', 'datasets/marriage_personality-traits/')
    data_X = pd.read_csv(thepath + "Xvals.csv")
    data_Y = pd.read_csv(thepath + "Yvals.csv")
    aff_data = pd.read_csv(thepath + "affinitymatrix.csv")
    nbx,nbk = data_X.shape
    nby,nbl = data_Y.shape
    A_k_l = aff_data.iloc[0:nbk,1:nbl+1].values

    if verbose:
        print(data_X.head())
        print(data_Y.head())
        print(tb.tabulate(A_k_l))
        
    return({'data_X': data_X,
            'data_Y': data_Y,
            'A_k_l': A_k_l})
           
def load_ChooSiow_data(nbCateg = 25):
    thepath = pkg_resources.resource_filename('mec', 'datasets/marriage-ChooSiow/')
    n_singles = pd.read_csv(thepath+'n_singles.txt', sep='\t', header = None)
    marr = pd.read_csv(thepath+'marr.txt', sep='\t', header = None)
    navail = pd.read_csv(thepath+'n_avail.txt', sep='\t', header = None)
    μhat_x0 = np.array(n_singles[0].iloc[0:nbCateg])
    μhat_0y = np.array(n_singles[1].iloc[0:nbCateg])
    μhat_xy = np.array(marr.iloc[0:nbCateg:,0:nbCateg])
    Nhat = 2 * μhat_xy.sum() + μhat_x0.sum() + μhat_0y.sum()    
    μhat_a = np.concatenate([μhat_xy.flatten(),μhat_x0,μhat_0y]) / Nhat # rescale the data so that the total number of individual is one

    return({'μhat_a':μhat_a, 
             'Nhat':Nhat,
             'nbx':nbCateg,
             'nby':nbCateg
             }) 

def load_Rust_data():
    thepath = pkg_resources.resource_filename('mec', 'datasets/dynamicchoice_Rust/')
    def getcleandata(name,nrow):
        filepath = thepath+name+'.asc'
        thearray =  pd.read_csv( filepath, delim_whitespace=True, header=None).to_numpy().reshape((nrow, -1), order='F')
        odometer1 = thearray[5, :]  # mileage at first replacement (0 if no replacement)
        odometer2 = thearray[8, :]  # mileage at second replacement (0 if no replacement)

        thearray = thearray[11:, :]
        
        replaced1 = (thearray >= odometer1) * (odometer1 > 0)  # replaced once
        replaced2 = (thearray >= odometer2) * (odometer2 > 0)  # replaced twice
        
        running_odometer = np.floor((thearray- odometer1 * replaced1 + (odometer1-odometer2)*replaced2 ) / 5000).astype(int)
        T,B = thearray.shape
        replact = np.array([[ (1 if (replaced1[t+1,b] and not replaced1[t,b]) or (replaced2[t+1,b] and not replaced2[t,b])  else 0) for b in range(B)]
                for t in range(T-1)]+
                        [[0 for b in range(B)]])
        increment = np.array([[ 0 for b in range(B)]]+
                            [[ running_odometer[t+1,b] - running_odometer[t,b] * (1-replact[t,b]) for b in range(B)] for t in range(T-1)])

        return np.block([[running_odometer.reshape((-1,1) ),replact.reshape((-1,1) ), increment.reshape((-1,1) )]])
    
    return np.vstack([getcleandata(name,nrow) for (name, nrow) in [ ('g870',36),('rt50',60),('t8h203',81),('a530875',128) ]])


