import numpy as np
from numba import njit
from pySODM.models.base import ODEModel

class packed_PFR(ODEModel):
    """
    A model of a packed-bed plug-flow reactor with axial dispersion in one dimension
    At the surface of the catalyst, the enzymatic esterification conversion of D-Glucose and Lauric acid into Glucose Laurate Ester and water takes place
    S + A <--> Es + W
    """

    states = ['C_F', 'C_S']
    parameters = ['delta_x', 'epsilon', 'u', 'rho_B','Vf_Ks', 'R_AS', 'R_AW', 'K_eq', 'R_Es']
    stratified_parameters = [['kL_a', 'D_ax'],[]]
    dimensions = ['species', 'x']

    @staticmethod
    @njit
    def integrate(t, C_F, C_S, delta_x, epsilon, u, rho_B, Vf_Ks, R_AS, R_AW, K_eq, R_Es, kL_a, D_ax):

        # Initialize derivatives
        dC_F = np.zeros(C_F.shape, dtype=np.float64)
        dC_S = np.zeros(C_S.shape, dtype=np.float64)

        # Reaction stochiometry
        stochiometry = [-1, -1, 1, 1]

        # dimension lengths
        N = C_F.shape[0]
        X = C_F.shape[1]

        # Loop over species: S, A, Es, W
        for i in range(N):
            # Loop over reactor length
            for j in range(1,X):
                # Evaluate the enzyme kinetic model
                v = (Vf_Ks*(C_S[0,j]*C_S[1,j] - (1/K_eq)*C_S[2,j]*C_S[3,j])/(C_S[1,j] + R_AS*C_S[0,j] + R_AW*C_S[3,j] + R_Es*C_S[2,j]))/60 # mmol/(s.g_catalyst)
                # Intermediate nodes
                if j < X-1:
                    dC_F[i,j] = (D_ax[i]/delta_x**2)*(C_F[i,j-1] - 2*C_F[i,j] + C_F[i,j+1]) - \
                                          (u/delta_x)*(C_F[i,j] - C_F[i,j-1]) + \
                                          (kL_a[i]/epsilon)*(C_S[i,j] - C_F[i,j])
                # Outlet boundary
                elif j == X-1:
                    dC_F[i,j] = (D_ax[i]/delta_x**2)*(C_F[i,j-1] - 2*C_F[i,j] + C_F[i,j-1]) - \
                                          (u/delta_x)*(C_F[i,j] - C_F[i,j-1]) + \
                                          (kL_a[i]/epsilon)*(C_S[i,j] - C_F[i,j])
                # Solid phase
                dC_S[i,j] = - (kL_a[i]/(1-epsilon))*(C_S[i,j] - C_F[i,j]) + rho_B*stochiometry[i]*v

        return dC_F, dC_S

class PPBB_model(ODEModel):
    """
    A model for the enzymatic esterification conversion of D-Glucose and Lauric acid into Glucose Laurate Ester and water
    S + A <--> Es + W
    """
    
    states = ['S','A','Es','W']
    parameters = ['c_enzyme', 'Vf_Ks', 'R_AS', 'R_AW', 'R_Es', 'K_eq']

    @staticmethod
    @njit
    def integrate(t, S, A, Es, W, c_enzyme, Vf_Ks, R_AS, R_AW, R_Es, K_eq):

        # Calculate rate
        v = c_enzyme*(Vf_Ks*(S*A - (1/K_eq)*Es*W)/(A + R_AS*S + R_AW*W + R_Es*Es))
       
        return -v, -v, v, v