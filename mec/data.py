def load_stigler_data(nbi = 9, nbj = 77, verbose=False):
    import pandas as pd
    #thepath = 'https://raw.githubusercontent.com/math-econ-code/mec_optim_2021-01/master/data_mec_optim/lp_stigler-diet/StiglerData1939.txt'
    thepath =data_file_path = pkg_resources.resource_filename('mec', 'datasets/StiglerData1939.txt')
    thedata = pd.read_csv(thepath , sep='\t')
    thedata = thedata.dropna(how = 'all')
    commodities = (thedata['Commodity'].values)[:-1]
    allowance = thedata.iloc[-1, 4:].fillna(0).transpose()
    nbi = min(len(allowance),nbi)
    nbj = min(len(commodities),nbj)
    if verbose:
        print('Daily nutrient content:')
        print(tabulate(thedata.head()))
        print('\nDaily nutrient requirement:')
        print(allowance)
    return({'N_i_j':thedata.iloc[:nbj, 4:(4+nbi)].fillna(0).to_numpy().T,
            'd_i':np.array(allowance)[0:nbi],
            'c_j':np.ones(len(commodities))[0:nbj],
            'names_i': list(thedata.columns)[4:(4+nbi)],
            'names_j':commodities[0:nbj]}) 
