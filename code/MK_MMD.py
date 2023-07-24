# multi-kernel maximum mean discrepancy
# cao bin, HKUST, China, binjacobcao@gmail.com
# free to charge for academic communication
# free to charge for academic communication

import numpy as np
from cvxopt import solvers, matrix 
from sklearn.gaussian_process.kernels import  RBF 

class MKMMD():
    def __init__(self, gamma_list=[1,1,1,1,1], kernel_num = 5):
        '''
        Our code is designed for educational purposes, 
        and to make it easier to understand, 
        we have implemented only the RBF (Radial Basis Function) kernel.
        
        This case focuses on solving the weights of kernels. 
        The estimation of length scales is crucial in kernel-based models,
        For further details on the method (length scales), please visit the following link: 
        [https://github.com/MaterialsInformaticsDemo/DAN/blob/main/code/MK_MMD.py].

        :param gamma_list: list of length scales for rbf kernels
        :param kernel_num: number of kernels in MK_MMD
        '''
        if len(gamma_list) != kernel_num: 
            print('please assign specific length scales for each rbf kernel')
        self.kernel_num = kernel_num
        kernel_list = []
        for i in range(kernel_num):
            kernel_list.append(RBF(gamma_list[i],"fixed"))
        self.kernel_list = kernel_list



    def predict(self, Xs, Xt,) :
        '''
        :param Xs: ns * m_feature, source domain data 
        :param Xt: nt * m_feature, target domain data

        return :
        the result of MK_MMD & weights of kernels 
        '''
        # cal weights for each rbf kernel
        # two rows above section 2.2 Empirical estimate of the MMD, asymptotic distribution, and test
        h_matrix = [] # 5 * 5 
        for i in range(self.kernel_num):
            h_k_vector = __fun(Xs, Xt, self.kernel_list[i], MMD = False, h_k_vector = True)
            h_matrix.append(h_k_vector)
        h_matrix = np.vstack(h_matrix)
        # cal the covariance matrix of h_matrix
        # Eq.(7)
        Q_k = np.cov(h_matrix)
        # cal the weights of kernels, Eq.(11)
        # vector η_k, Eq.(2)
        η_k = []
        for k in range(self.kernel_num):
            MMD = __fun(Xs, Xt, self.kernel_list[k], MMD = True, h_k_vector = False)
            η_k.append(MMD)

        # solve the standard quadratic programming problem 
        # see : https://github.com/Bin-Cao/KMMTransferRegressor/blob/main/KMMTR/KMM.py
        P = 2 * matrix(Q_k + 1e-5 * np.eye(self.kernel_num)) # λm = 1e-5 
        q = matrix(np.zeros((self.kernel_num,1)))
        G = matrix(np.eye(self.kernel_num))
        # A = η_k 
        A = matrix(η_k)
        b=matrix(1.)
        h=matrix(0.)
        # P is 5 * 5
        # q is 5 * 1
        # G is 1 * 5
        # A is 1 * 5
        # b = 1, h =0
        solvers.options['show_progress'] = True
        sol = solvers.qp(P,q,G,h,A,b)
        beta = sol['x']
        
        MK_MMD = np.array(η_k) * np.array(beta).reshape(-1,1)

        return MK_MMD, np.array(beta)
        
def __fun(Xs, Xt, kernel, MMD = True, h_k_vector = False):
    if MMD == True:
        # cal MMD for one rbf kernel
        # Eq.(1) in paper
        EXX_= kernel(Xs,Xs)
        EYY_= kernel(Xt,Xt)
        EYX_= kernel(Xt,Xs)
        EXY_= kernel(Xs,Xt)
        MMD = np.array(EXX_).mean() + np.array(EYY_).mean() - np.array(EYX_).mean() - np.array(EXY_).mean()
    else: MMD = None

    if h_k_vector == True:
        # cal vector h_k(x,x',y,y'), contains m**2*n**2 terms
        # between Eq.(1) and Eq.(2)
        # k(x, x') is the element of matrix EXX_
        # k(y, y') is the element of matrix EYY_
        # k(x, y') and k(x', y) are the element of matrix EXY_
        ns, nt = len(Xs), len(Xt)
        combin_ns = generate_combinations(ns)
        combin_nt = generate_combinations(nt)
        h_k_vector = []
        for x in range(len(combin_ns)):
            for y in range(len(nt)):
                h_k = kernel(Xs[combin_ns[0]],Xs[combin_ns[1]]) + kernel(Xt[combin_nt[0]],Xt[combin_nt[1]]) 
                - kernel(Xs[combin_ns[0]],Xt[combin_nt[1]]) - kernel(Xs[combin_ns[1]],Xt[combin_nt[0]])
                h_k_vector.append(h_k)
        h_k_vector = np.array(h_k_vector)
    else: h_k_vector = None

    return MMD, h_k_vector

    

def generate_combinations(n):
    # Cn^2
    combinations = []
    for i in range(n):
        for j in range(i, n):
            combinations.append((i, j))
    return combinations
