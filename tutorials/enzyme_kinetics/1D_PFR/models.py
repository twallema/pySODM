import numpy as np
from pySODM.models.base import ODEModel

class packed_PFR(ODEModel):
    """
    A model of a packed-bed plug-flow reactor with axial dispersion in one dimension
    At the surface of the catalyst, the enzymatic esterification conversion of D-Glucose and Lauric acid into Glucose Laurate Ester and water takes place
    S + A <--> Es + W
    """

    state_names = ['S', 'A', 'Es', 'W']
    parameter_names = ['epsilon', 'kL_a', 'D_ax', 'delta_x', 'u', 'rho_B',
                       'Vf_Ks', 'R_AS', 'R_AW', 'K_eq', 'K_iEs', 'R_Es', 'K_W']
    stratification_names = ['phase', 'x']

    @staticmethod
    def integrate(t, S, A, Es, W, epsilon, kL_a, D_ax, delta_x, u, rho_B, Vf_Ks, R_AS, R_AW, K_eq, K_iEs, R_Es, K_W):

        # Initialize derivatives
        dS = np.zeros(S.shape, dtype=np.float64)
        dA = np.zeros(A.shape, dtype=np.float64)
        dEs = np.zeros(Es.shape, dtype=np.float64)
        dW = np.zeros(W.shape, dtype=np.float64)

        # Place them in a list to ease further handling
        derivatives = [dS, dA, dEs, dW]
        species = [S, A, Es, W]

        # Define stochiometry
        stochiometry = [-1, -1, 1, 1]

        # Loop over species
        for i in range(len(['S', 'A', 'Es', 'W'])):
            # Loop over spatial axes
            for j in range(1,S.shape[1]):
                # Evaluate the enzyme kinetic model
                v = Vf_Ks*(S[1,j]*A[1,j] - (1/K_eq)*Es[1,j]*W[1,j])/(A[1,j] + R_AS*S[1,j] + R_AW*W[1,j] + R_AS*S[1,j]*Es[1,j]/K_iEs + R_Es*Es[1,j] + R_Es*W[1,j]*Es[1,j]/K_W)
                # Liquid phase
                C = species[i]
                # Intermediate nodes
                if j < S.shape[1]-1:
                    derivatives[i][0,j] = (D_ax[i]/delta_x**2)*(C[0,j-1] - 2*C[0,j] + C[0,j+1]) - \
                                          (u/delta_x)*(C[0,j] - C[0,j-1]) + \
                                          (kL_a[i]/epsilon)*(C[1,j] - C[0,j])
                # Outlet boundary
                elif j == S.shape[1]-1:
                    derivatives[i][0,j] = (D_ax[i]/delta_x**2)*(C[0,j-1] - 2*C[0,j] + C[0,j-1]) - \
                                          (u/delta_x)*(C[0,j] - C[0,j-1]) + \
                                          (kL_a[i]/epsilon)*(C[1,j] - C[0,j])
                # Solid phase
                derivatives[i][1,j] = - (kL_a[i]/(1-epsilon))*(C[1,j] - C[0,j]) + rho_B*stochiometry[i]*v/10

        # Unpack derivatives
        dS, dA, dEs, dW = derivatives

        return dS, dA, dEs, dW

class PPBB_model(ODEModel):
    """
    A model for the 
    S + A <--> Es + W
    """
    
    state_names = ['S','A','Es','W']
    parameter_names = ['c_enzyme', 'Vf_Ks', 'R_AS', 'R_AW', 'R_Es', 'K_eq', 'K_W', 'K_iEs']

    @staticmethod
    def integrate(t, S, A, Es, W, c_enzyme, Vf_Ks, R_AS, R_AW, R_Es, K_eq, K_W, K_iEs):

        # Calculate rate
        v = c_enzyme*(Vf_Ks*(S*A - (1/K_eq)*Es*W)/(A + R_AS*S + R_AW*W + R_AS*S*Es/K_iEs + R_Es*Es + R_Es*W*Es/K_W))
       
        return -v, -v, v, v