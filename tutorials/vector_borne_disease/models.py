import numpy as np
from pySODM.models.base import ODEModel

class SIR_SI(ODEModel):
    """
    An age-stratified SIR model for humans, an unstratified SI model for a disease vector (f.i. mosquito)
    """

    state_names = ['S', 'I', 'R', 'S_v', 'I_v']
    parameter_names = ['gamma']
    parameter_stratified_names = ['alpha']
    stratification_names = ['age_group']
    state_stratifications = [['age_group'],['age_group'],['age_group'],[],[]]


    @staticmethod
    def integrate(t, S, I, R, S_v, I_v, alpha, gamma):

        # Calculate total mosquito population
        N_v = S_v + I_v
        # Calculate human differentials
        dS = -alpha*(I_v/N_v)*S
        dI = alpha*(I_v/N_v)*S - 1/gamma*I
        dR = 1/gamma*I
        # Calculate mosquito differentials
        dS_v = -np.sum(alpha*(I_v/N_v)*S)
        dI_v = np.sum(alpha*(I_v/N_v)*S)

        return dS, dI, dR, dS_v, dI_v