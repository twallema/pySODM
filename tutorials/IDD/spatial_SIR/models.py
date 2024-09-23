"""
This script contains implementations of a age- and space stratified SIR model.
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import numpy as np
from pySODM.models.base import ODE, JumpProcess

###################
## Deterministic ##
###################

class spatial_ODE_SIR(ODE):
    """
    SIR model with a spatial stratification
    """
    
    states = ['S','I','R']
    parameters = ['beta','gamma', 'f_v', 'N', 'M']
    dimensions = ['age', 'location']

    @staticmethod
    def integrate(t, S, I, R, beta, gamma, f_v, N, M):

        # compute total population 
        T = S + I + R

        # compute visiting populations
        T_v = matmul_2D_3D_matrix(T, M) # M can  be of size (n_loc, n_loc) or (n_loc, n_loc, n_age), representing a different OD matrix in every age group
        S_v = matmul_2D_3D_matrix(S, M)
        I_v = matmul_2D_3D_matrix(I, M)

        # compute number of new infections on home patch and visited patch
        dI_h = beta * S * np.transpose(matmul_2D_3D_matrix(np.transpose(I/T), (1-f_v)*N)) # N can  be of size (n_age, n_age) or (n_age, n_age, n_loc), representing a different contact matrix in every spatial patch 
        dI_v = beta * S_v * np.transpose(matmul_2D_3D_matrix(np.transpose(I_v/T_v), f_v*N))

        # distribute the number of new infections on visited patch to the home patch 
        dI_v = S * np.transpose(np.atleast_2d(M) @ np.transpose(dI_v/S_v))

        # Calculate differentials
        dS = - (dI_h + dI_v)
        dI = (dI_h + dI_v) - 1/gamma*I
        dR = 1/gamma*I

        return dS, dI, dR

    # @staticmethod
    # def integrate(t, S, I, R, beta, gamma, f_v, N, M):

    #     # compute total population
    #     T = S + I + R

    #     # compute visiting populations
    #     I_v = I @ M
    #     T_v = T @ M

    #     #  compute force of infection
    #     l = beta * (np.einsum ('lj, il -> ij', I/T, (1-f_v)*N) + np.einsum ('jk, lk, il -> ij', M, I_v/T_v, f_v*N))

    #     # calculate differentials
    #     dS = - l * S
    #     dI = l * S - 1/gamma*I
    #     dR = 1/gamma*I

    #     return dS, dI, dR

###################
### Stochastic ###
###################

class spatial_TL_SIR(JumpProcess):
    """
    SIR stochastic model with a spatial stratification
    """
    states = ['S', 'S_work','I','R']
    parameters = ['beta','gamma', 'f_v', 'N', 'M']
    dimensions = ['age', 'location']


    @staticmethod
    def compute_rates(t, S, S_work, I, R, beta, gamma, f_v, N, M):

        # calculate total population 
        T = S + I + R

        # compute visiting populations
        T_v = matmul_2D_3D_matrix(T, M) # M can  be of size (n_loc, n_loc) or (n_loc, n_loc, n_age), representing a different OD matrix in every age group
        I_v = matmul_2D_3D_matrix(I, M)

        # create a size "dummy"
        size_dummy = np.ones(S.shape, np.float64)

        rates = {

            'S': [beta * np.transpose(matmul_2D_3D_matrix(np.transpose(I/T), (1-f_v)*N))], # 
            'S_work': [beta * np.transpose(matmul_2D_3D_matrix(np.transpose(I_v/T_v), f_v*N))],
            'I': [size_dummy*(1/gamma)], # 

            }
        return rates

    @ staticmethod
    def apply_transitionings(t, tau, transitionings, S, S_work, I, R, 
                             beta, f_v, gamma, 
                             N, M):
        
        # distribute the number of new infections on visited patch to the home patch 
        S_work_to_home = S * np.transpose(np.atleast_2d(M) @ np.transpose(transitionings['S_work'][0]/S_work))
        # the resulting matrix is an N x M matrix with each element of row n, column m representing the total number of people infected from work that need to be returned to age-group n, location m. 

        # Calculate new states
        S_new = S - transitionings['S'][0] - S_work_to_home[0]
        S_work_new =  matmul_2D_3D_matrix(S_new, M)
        I_new = I + transitionings['S'][0] + S_work_to_home[0] - transitionings['I'][0]
        R_new = R + transitionings['I'][0]
        
        return(S_new,S_work_new, I_new, R_new)
    
# helper function
def matmul_2D_3D_matrix(X, W):
    """
    Computes the product of a 2D matrix (size n x m) and a 3D matrix (size m x m x n) as an n-dimensional stack of (1xm) and (m,m) products.

    input
    =====
    X: np.ndarray
        Matrix of size (n,m).
    W : np.ndarray
        2D or 3D matrix:
        - If 2D: Shape (m, m). Expanded to size (m, m, n).
        - If 3D: Shape (m, m, n).
          Represents n stacked (m x m) matrices.

    output
    ======
    X_out : np.ndarray
        Matrix product of size (n, m). 
        Element-wise equivalent operation: O_{ij} = sum_{l} [ s_{il} * w_{lji} ]
    """
    W = np.atleast_3d(W)
    return np.einsum('ik,kji->ij', X, np.broadcast_to(W, (W.shape[0], W.shape[0], X.shape[0])))
