"""
This script contains a spatially-explicit SIR model.
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import numpy as np

###################
## Deterministic ##
###################

# import the ODE class
from pySODM.models.base import ODE

# helper function
def matmul_2D_3D_matrix(X, W):
    """
    Computes the matrix product of a 2D matrix (size n x m) and a 3D matrix (size m x m x n).

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
    return np.einsum('ik,kji->ij', X, np.broadcast_to(np.atleast_3d(W), (W.shape[0], W.shape[0], X.shape[0])))

# Define the model equations
class spatial_ODE_SIR(ODE):
    """
    SIR model with a spatial stratification
    """
    
    states = ['S','I','R']
    parameters = ['beta','gamma']
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
        dI_h = beta * S * np.transpose(matmul_2D_3D_matrix(np.transpose(I/T), N)) # N can  be of size (n_age, n_age) or (n_age, n_age, n_loc), representing a different contact matrix in every spatial patch 
        dI_v = beta * S_v * np.transpose(matmul_2D_3D_matrix(np.transpose(I_v/T), N))

        # distribute the number of new infections on visited patch to the home patch 
        dI_h = S * np.transpose(M @ np.transpose(dI_h/S_v))

        # Calculate differentials
        dS = - (dI_h + dI_v)
        dI = (dI_h + dI_v) - 1/gamma*I
        dR = 1/gamma*I

        return dS, dI, dR
