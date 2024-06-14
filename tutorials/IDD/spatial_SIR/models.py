"""
This script contains a spatially-explicit SIR model.
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import numpy as np
from pySODM.models.base import ODE, JumpProcess

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
        Element-wise equivalent operation: O_{ij} = \sum_{l} [ s_{il} * w_{lji} ]
    """
    W = np.atleast_3d(W)
    return np.einsum('ik,kji->ij', X, np.broadcast_to(W, (W.shape[0], W.shape[0], X.shape[0])))

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
