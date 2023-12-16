import numpy as np
import mec.data

def test_mec_lp_LP():
    from mec.lp import LP   
    data = mec.data.load_stigler_data(verbose = True)
    dietLP = LP(data['N_i_j'].T,data['c_j'],data['d_i'])
    dietLP.gurobi_solve(verbose=0)
    return

def test_mec_lp_Dictionary():
    from mec.lp import Dictionary
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
    [N_i_j,minallowance_i,unitcost_j,_,_] = mec.data.load_stigler_data(7,5).values()
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
def test_mec_ot_OTProblem(limited=False):
    from mec.ot import OTProblem
    [data_X,data_Y,A_k_l] =  mec.data.load_DupuyGalichon_data().values()
    sdX,sdY = data_X.std().values, data_Y.std().values
    mX,mY = data_X.mean().values, data_Y.mean().values
    feats_x_k, feats_y_l = ((data_X-mX)/sdX).values, ((data_Y-mY)/sdY).values
    nbx,nbk = feats_x_k.shape
    nby,nbl = feats_y_l.shape
    Φ_x_y = feats_x_k @ A_k_l @ feats_y_l.T
    if limited:
        Φ_x_y = Φ_x_y[0:10,0:10]
    marriageEx = OTProblem(Φ_x_y)
    (μ_x_y,u_x,v_y) = marriageEx.solve_full_lp()
    print('Man 0 matches with woman '+str(np.argwhere(μ_x_y[0,:] != 0)[0][0])+'.'  )
    (μ_x_y,u_x,v_y) = marriageEx.solve_partial_lp()
    print(μ_x_y.sum())
    u_x,v_y = marriageEx.solve_dual_partial_multiobj_lp(favor_Y=True)
    print(u_x.min(),v_y.min())
    if limited:
        nbx,nby = 10,8
    else:
        nbx,nby = 50,30
    marriage_ex = OTProblem(Φ_x_y[:nbx,:nby],np.ones(nbx) / nbx, np.ones(nby) / nby)
    nrow , ncol = min(8, nbx) , min(8, nby)
    marriage_ex.solve_full_lp()
    marriage_ex.logdomainIPFP_with_LSE_trick(0.5)
    marriage_ex.solveGLM(0.5 )
    marriage_ex.matrixIPFP(0.5 )

    return
    
def test_mec_ot_TULogit():
    def build_surplus(nbx,nby):
        xs, ys = np.repeat(range(1,nbx+1),nby).reshape(nbx,nby)/nbx, np.repeat(range(1,nby+1),nbx).reshape(nbx,nby).T/nby
        phi1_xy = -((xs-ys)**2).flatten()
        phimat = np.vstack([phi1_xy,
                            -((xs-ys)**2 * ((xs+ys)/2)**2).flatten(),
                             -( (xs-ys)**2 * ((xs+ys-2)/2)**2).flatten(),
                            -( (xs-ys)**2 *(xs+ys-1)**2).flatten()
                           ]).T
        phimat_mean = phimat.mean(axis = 0 , keepdims=True)
        phimat_stdev = phimat.std(axis = 0 , keepdims=True)
        φ_xy_k = np.hstack([np.ones((nbx*nby,1)) ,(phimat - phimat_mean)/phimat_stdev]  )
        _,nbk = φ_xy_k.shape
        return np.vstack( [φ_xy_k ,np.zeros( (nbx,nbk) ),np.zeros( (nby,nbk) )])

    from mec.ot import TUlogit
    [cs_μhat_a, cs_Nhat,cs_nbx,cs_nby] = mec.data.load_ChooSiow_data().values()
    choo_siow_mkt = TUlogit(cs_nbx,cs_nby,build_surplus(cs_nbx,cs_nby),cs_μhat_a)  
    uv_GS_diy,λ_GS_diy_k = choo_siow_mkt.fit_diy(10000,tol = 1e-8)
    uv_GS_GLM, λ_GS_GLM_k = choo_siow_mkt.fit_glm()
    print('λ_GS_GLM_k = '+str(λ_GS_GLM_k))
    print('λ_GS_diy_k = '+str(λ_GS_diy_k))
    print(choo_siow_mkt.assess(np.append(uv_GS_GLM, λ_GS_GLM_k)))
    print(choo_siow_mkt.assess(np.append(uv_GS_diy, λ_GS_diy_k)))
    choo_siow_mkt.isCoercive()
    return
########################################################
########################################################
def test_mec_gt_Matrix_game():
    from mec.gt import Matrix_game
    Phi_i_j = np.array([[50, 100, 93.8], 
                    [93.9, 60.0, 82.8], 
                    [97.6, 100, 73.2]])
    thegame = Matrix_game(Phi_i_j)
    print(thegame.BRI(1) ,thegame.BRJ(0) )
    thegame.compute_eq()
    thegame.minimax_LP()
    thegame.minimax_CP()



def test_mec_gt_Bimatrix_game():
    from mec.gt import Bimatrix_game
    penalty_data = np.array([[50, 100, 93.8], [93.9, 60.0, 82.8], [97.6, 100, 73.2]])

    penalty_zero_sum = Bimatrix_game(A_i_j = penalty_data, B_i_j = 100 - penalty_data)
    penalty_nonzero_sum = Bimatrix_game(A_i_j = penalty_data,
                                    B_i_j = np.array([[150, 100, 100], [100, 150, 100], [100, 100, 150]]) - penalty_data)
    penalty_zero_sum.mangasarian_stone_solve()
    penalty_nonzero_sum.lemke_howson_solve()

########################################################
########################################################
########################################################

def test_mec_lp():
    test_mec_lp_LP()
    test_mec_lp_Dictionary()
    test_mec_lp_Tableau()
    return


def test_mec_ot(limited=False):
    test_mec_ot_OTProblem(limited)
    test_mec_ot_TULogit()
    
    return
    
def test_mec_gt():
    test_mec_gt_Matrix_game()
    test_mec_gt_Bimatrix_game()
    
    return

def test_mec(limited=False):
    print('Linear programming tests:\n'+'*'*30)
    test_mec_lp()
    print('*'*30+'\nOptimal transport tests:\n'+'*'*30)
    test_mec_ot(limited)
    print('*'*30+'\nGame theory tests:\n'+'*'*30)
    test_mec_gt()
    print('*'*30+'\n'+'*'*30+'\nAll tests completed successfully.')
    return
    
