import numpy as np
import pkg_resources
import pandas as pd
import tabulate as tb


def load_stigler_data(nbi = 9, nbj = 77, verbose=False):

    thepath = pkg_resources.resource_filename('mec', 'datasets/stigler-diet/StiglerData1939.txt')
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
            'b_i':np.array(allowance)[0:nbi],
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
    thepath = pkg_resources.resource_filename('mec', 'datasets/marriage_personality-traits/')
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
    """
    Returns dynamic discrete choice data based on Rust's (Econometrica 1987) original dataset

    output:
        a three column array whose:
         - first column is the milage range (per 5,000 miles brackets)
         - second column is the decision to replace (1) or not (0)
         - third column is number of additional mileage brackets after running one period 

    """
    thepath = pkg_resources.resource_filename('mec', 'datasets/dynamicchoice_Rust/')
    def getcleandata(name,nrow):
        filepath = thepath+name+'.asc'
        thearray = np.genfromtxt(filepath, delimiter=None, dtype=float).reshape((nrow, -1), order='F')
        #thearray =  pd.read_csv( filepath, sep='\s+', header=None).to_numpy().reshape((nrow, -1), order='F')
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



def load_blp_car_data(fast = True, check_pyblp_compatibility=False):
    """
    Returns the data used by Berry, Levinsohn and Pakes (1999)
    
    output:
        a pandas dataframe with product-level observations
    """

    if fast:
        thepath = pkg_resources.resource_filename('mec', 'datasets/demand_blp/preloaded/')
        prods = pd.read_csv(thepath+"prod.csv", index_col=0)
        agent_data = pd.read_csv(thepath+"agents.csv", index_col=0)
        return prods,agent_data

    thepath = pkg_resources.resource_filename('mec', 'datasets/demand_blp/blp_1999/')
    from mec.blp import create_blp_instruments

    column_names = [
        'name',       # name: Model name
        'car_ids',    # car_ids: Identification number
        'market_ids', # market_ids: Year
        'cy',         # cy: Number of cylinders
        'dr',         # dr: Drive type (2 or 4)
        'at',         # at: Automatic transmission
        'ps',         # ps: Power steering
        'air',        # air: Air conditioning
        'drv',        # drv: Drive type (front wheel drive)
        'prices',      # price: Price in dollars
        'wght',       # wght: Weight in pounds
        'dom',        # dom: Domestic or imported
        'disp',       # disp: Displacement
        'hp',         # hp: Horsepower
        'lngth',      # lngth: Length in inches
        'wdth',       # wdth: Width in inches
        'wb',         # wb: Wheelbase in inches
        'mpg',        # mpg: Miles per gallon
        'sales',      # sales: Sales
        'firm_ids',   # fid: Firm ID
        'eu',         # eu: European model
        're',         # re: Reliability index
        'sales*',     # unclear (probably related to foreign sales)
        'dfi',        # dfi: Transplant production in the U.S.
        'ci'          # ci: captive imports against the voluntary export restraint (ver)
    ]
    # load data files
    prods = pd.read_csv(thepath+'panel6.asc', sep='\s+', names=column_names, header=None)
    otherdf = pd.read_csv( thepath + 'otherdat.asc', 
                          sep='\s+', 
                          names= ['year','size', 'meanly' ,'cpi' ,'gasprice', 'rmat1','rmat2', 
                                  'nvec','xx1','xx2','xx3','xx4','xx5'] ,
                          header=None)
    otherdf3 = pd.read_csv( thepath + 'otherdat3.asc', 
                           sep='\s+', 
                           names= ['year','size', 'meanly' ,'cpi' ,'gasprice', 'rmat1','rmat2', 
                                   'nvec','uswage','gwage','jwage','dm', 'yen', 'gnp' , 'primer'],
                           header=None).head(20)
    erates = pd.read_csv( thepath + 'lagged_erate.asc', sep='\s+', names=['yenlag','dmlag'] ,header=None)

    unobs = pd.read_csv(thepath+'../gentzkow-shapiro/unobs_pub.csv').dropna(axis=1, how='all')
    
    
    # convert 1990 displacement from liters to cubic inches
    prods.loc[prods['market_ids'] == 90, 'disp'] = prods.loc[prods['market_ids'] == 90, 'disp'] * 61.02
    #
    # compute mile per dollar = mile per gallon / dollar per gallon
    adjustment_dic = {otherdf['year'][i]: otherdf['cpi'][i] /( 100*  otherdf['gasprice'][i] ) for i,_ in otherdf.iterrows() }
    mpd = np.array([prods['mpg'][i] * adjustment_dic[prods['market_ids'][i]] / 10 for i,_ in prods.iterrows() ])
    prods['mpd'] = mpd
    #
    otherdf3['yenlag'] = erates['yenlag']
    otherdf3['dmlag'] = erates['dmlag']
    #
    # create market shares
    size_dic =  {otherdf3['year'][i]: otherdf3['size'][i]  for i,_ in otherdf3.iterrows() }
    prods['shares'] = np.array([prods['sales'][i] / size_dic[prods['market_ids'][i]] / 1000 for i,_ in prods.iterrows() ])

    # deflate prices and rescale various quantities
    cpi_dic = {otherdf['year'][i]: otherdf['cpi'][i] for i,_ in otherdf.iterrows() }
    prods['prices'] = [prods['prices'][i] / (cpi_dic[prods['market_ids'][i]] / 100) / 1000 for i,_ in prods.iterrows() ]
    prods['sales'] = prods['sales'] / 1000
    prods['wght'] =prods['wght'] / 1000
    prods['hp'] =prods['hp'] / 100
    prods['disp'] =prods['disp'] / 100
    prods['lngth'] =prods['lngth'] / 100
    prods['wdth'] =prods['wdth'] / 100
    prods['wb'] =prods['wb'] / 100
    prods['mpg'] =prods['mpg'] / 10
    ###################################
    # build clustering ids
    prods['clustering_ids'] = [prods['name'][i]+str(prods['market_ids'][i]) for i,_ in prods.iterrows() ]
    for name in prods['name'].unique():
        chars = prods[prods['name'] == name ][['market_ids','car_ids','hp','lngth','wdth','wb']].to_numpy()
        iref = 0
        for i in range(chars.shape[0]):
            if (2* np.abs( chars[i,2:] - chars[iref,2:]) /  (chars[i,2:]+chars[iref,2:]) ).max() >= 0.1:
                iref = i
            else:
                prev_model_cid = prods[(prods['car_ids'] == chars[iref,1]) ]['clustering_ids'].iloc[0]
                prods.loc[(prods['car_ids'] == chars[i,1]), 'clustering_ids']= prev_model_cid

    if check_pyblp_compatibility:
        # sometimes there are multiple products with the same name, same year; it is not clear how BLP treat that but the following 
        # ensures compatibility with pyblp by adjusting them manually
        cids_to_change = {1540: 'TYCORO71', 1542: 'TYCORO71', 1582: 'TYCORO71', 560: 'MCMONT73', 1084: 'OD9878', 1230: 'OD9878', 
            1854: 'TYCORO78', 1384: 'OD9880', 1867: 'AD500080', 2410: 'TYCORO81', 2424: 'AD500081', 2451: 'MB380S82',
            2511: 'MB380S82', 3351: 'DGCOLT85', 3434: 'PTGRAN84', 3471: 'MB380S85', 3648: 'DGCOLT85', 3954: 'DGCOLT86',
            4247: 'DGCOLT87', 4316: 'PTGRAN86',4485: 'SA900089', 5517: 'MB560S90', 5563: 'SA900090', 5590: 'PS911C90'}

        for i in cids_to_change.keys():
            prods.loc[(prods['car_ids'] == i), 'clustering_ids']= cids_to_change[i]
    #
    prods['market_ids'] = 1900+prods['market_ids']
    prods['region']=[ 'US' if (prods['dom'][i]==1) else (  'EU'  if (prods['eu'][i]==1) else 'JP') for i,_ in prods.iterrows()]
    #prods['verdum']= 1 - prods['dom'] - prods['eu'] - pd.Series(np.where(prods['firm_ids'] == 21 , 1, 0)) - prods['dfi'] + prods['ci']
    prods['hpwt'] = prods['hp'] / prods['wght']
    prods['space']= prods['lngth'] * prods['wdth']
    prods['trend'] = prods['market_ids'] - prods['market_ids'][0]
    #
    mkts_firms_prods = prods[['market_ids', 'firm_ids', 'car_ids']].to_numpy()
    instr_vals = prods[['hpwt', 'air', 'mpd']].to_numpy()
    prods[['demand_instruments'+str(k) for k in range(8)] ] = create_blp_instruments(instr_vals,mkts_firms_prods,include_ones=1, include_arguments=False)
    instr_vals =pd.DataFrame([np.log(prods['hpwt']), prods['air'], np.log(prods['mpg']),np.log(prods['space']) ]).T.to_numpy()
    prods[['supply_instruments'+str(k) for k in range(10)] ] = create_blp_instruments(instr_vals,mkts_firms_prods,include_ones=True , include_arguments=False)
    instr_vals = prods['trend'].to_numpy().reshape((-1,1))
    prods[['supply_instruments'+str(k) for k in [10,11] ] ] = create_blp_instruments(instr_vals,mkts_firms_prods,include_ones=0, include_arguments = False)
    prods['supply_instruments11'] = prods['mpd']
    # now, prepare agent data
    mean_incomes_t = otherdf3['meanly'].to_numpy()
    sd_incomes = 1.72
    T,I = len(mean_incomes_t), len(unobs)
    O_t,O_i = np.zeros(T),np.zeros(I)
    income_random_i = unobs['income'].to_numpy()
    income_ti = np.exp( mean_incomes_t[:,None]+ sd_incomes * income_random_i[None,:] ).flatten()
    year_ti = (np.arange(1971,1991)[:,None]+O_i[None,:]).flatten()
    weights_ti = (O_t[:,None] + unobs['weight'].to_numpy()[None,:]/200).flatten()
    nodes_ti_k = (O_t[:,None,None] + unobs[ ['const','hpwt','air','mpd','space']].to_numpy()[None,:,:]).reshape((T*I,-1))
    agent_data=pd.DataFrame()
    agent_data[ ['market_ids','weights']+ ['nodes' +str(i) for i in [0,1,2,3,4] ] + ['income'] ] = np.block([[year_ti[:,None],weights_ti[:,None],nodes_ti_k,income_ti[:,None] ]])
    agent_data['market_ids']=agent_data['market_ids'].astype(int) 

    # check consistency with pyblp
    if check_pyblp_compatibility:
        import pyblp
        # check if product data coincide
        product_data = pd.read_csv(pyblp.data.BLP_PRODUCTS_LOCATION)
        list_problems = []
        for thecol in product_data.columns:
            if pd.to_numeric(prods[thecol], errors='coerce').notna().all():
                same = ( (np.abs(product_data[thecol]-prods[thecol])  ) <= 1e-5 * np.abs(product_data[thecol]+prods[thecol]) ) .all()  
            else:
                same = (product_data[thecol]== prods[thecol]).all()
            if not same:
                list_problems.append(thecol)
        if list_problems:
            print("Discrepancy in product data with pyblp for the following variables: "+', '.join(list_problems))
        else:
            print("Product data matches pyblp.")
        
        # now check if agent data coincide
        agent_data_pyblp = pd.read_csv(pyblp.data.BLP_AGENTS_LOCATION)
        if (agent_data - agent_data_pyblp).to_numpy().max()>1e-10:
            print('*** Problem ***. Agent data does not match pyblp.')
        else:
            print('Agent data matches pyblp.')

    return prods,agent_data
