def test_mec_lp_LP():
    from mec.lp import LP   
    from mec.data import load_stigler_data    
    data = load_stigler_data(verbose = True)
    dietLP = LP(data['N_i_j'].T,data['c_j'],data['d_i'])
    dietLP.gurobi_solve(verbose=0)
    return

def test_mec_lp_Dictionary():
    from mec.lp import Dictionary
    from mec.data import load_stigler_data
    import numpy as np
    dictionary_example = Dictionary(A_i_j = np.array([[2, 1], [1, 2]]),
                                d_i = np.array([2,2]),
                                c_j = np.array([1,1]),
                                slack_var_names_i = ['s_1', 's_2'],
                                decision_var_names_j = ['x_1', 'x_2']
                                )
    dictionary_example.display()
    path_example = [dictionary_example.primal_solution(verbose=1)]
    dictionary_example.plot2d(path_example)
    dictionary_example.simplex_loop(verbose=3)
    [N_i_j,minallowance_i,unitcost_j,_,_] = load_stigler_data(7,5).values()
    stigler_dictionary = Dictionary(N_i_j.T, unitcost_j , minallowance_i,
                             ['s_' + str(j) for j in range(N_i_j.shape[1])],
                             ['π_' + str(i) for i in range(N_i_j.shape[0])])
    stigler_dictionary.simplex_loop(verbose=2)[1]
    return
    
def test_mec_lp_Tableau():
    from mec.lp import Tableau
    from mec.data import load_stigler_data
    [N_i_j,minallowance_i,unitcost_j,_,_] = load_stigler_data().values()
    stigler_tableau = Tableau(N_i_j.T, unitcost_j , minallowance_i,
                             ['s_' + str(j) for j in range(N_i_j.shape[1])],
                             ['π_' + str(i) for i in range(N_i_j.shape[0])])

    stigler_tableau.simplex_solve()[1]
    return
########################################################
########################################################
########################################################
def test_mec_ot_OTProblem():
    from mec.ot import OTProblem
    from mec.data import load_DupuyGalichon_data  
    data = load_DupuyGalichon_data()
    [data_X,data_Y,A_k_l] = list(data)
    sdX,sdY = data_X.std().values, data_Y.std().values
    mX,mY = data_X.mean().values, data_Y.mean().values
    feats_x_k, feats_y_l = ((data_X-mX)/sdX).values, ((data_Y-mY)/sdY).values
    nbx,nbk = feats_x_k.shape
    nby,nbl = feats_y_l.shape
    Φ_x_y = feats_x_k @ A_k_l @ feats_y_l.T
    marriageEx = OTProblem(Φ_x_y)
    (μ_x_y,u_x,v_y) = marriageEx.solve_full_lp()
    print('Man 0 matches with woman '+str(np.argwhere(μ_x_y[0,:] != 0)[0][0])+'.'  )
    (μ_x_y,u_x,v_y) = marriageEx.solve_partial_lp()
    print(μ_x_y.sum())
    u_x,v_y = marriageEx.solve_dual_partial_multiobj_lp(favor_Y=True)
    print(u_x.min(),v_y.min())

    return
    


def test_mec_lp():
    test_mec_lp_LP()
    test_mec_lp_Dictionary()
    test_mec_lp_Tableau()
    return


def test_mec_ot():
    test_mec_ot_OTProblem
    
    return

def test_mec():
    test_mec_lp()
    test_mec_ot()
    test_mec_gt()
    return
    