import numpy as np
from pySODM.models.base import SDEModel, ODEModel

class SDE_SIR_SI(SDEModel):
    """
    A stochastic, age-stratified SIR model for humans, an unstratified SI model for a disease vector (f.i. mosquito)
    """

    states = ['S', 'I', 'R', 'S_v', 'I_v']
    parameters = ['beta', 'gamma']
    stratified_parameters = ['alpha']
    dimensions = ['age_group']
    state_dimensions = [['age_group'],['age_group'],['age_group'],[],[]]

    @staticmethod
    def compute_rates(t, S, I, R, S_v, I_v, alpha, beta, gamma):

        # Calculate total mosquito population
        N = S + I + R
        N_v = S_v + I_v

        rates = {
            'S': [alpha*(I_v/N_v),],
            'I': [(1/gamma)*np.ones(N.shape),],
            'S_v': [np.sum(alpha*(I/N))*np.ones(N_v.shape), (1/beta)*np.ones(N_v.shape),],
            'I_v': [(1/beta)*np.ones(N_v.shape),]
        }

        return rates

    @staticmethod
    def apply_transitionings(t, tau, transitionings, S, I, R, S_v, I_v, alpha, beta, gamma):

        S_new = S - transitionings['S'][0]
        I_new = I + transitionings['S'][0] - transitionings['I'][0]
        R_new = R + transitionings['I'][0]
        S_v_new = S_v - transitionings['S_v'][0] + transitionings['I_v'][0]
        I_v_new = I_v + transitionings['S_v'][0] - transitionings['I_v'][0]

        return S_new, I_new, R_new, S_v_new, I_v_new

class ODE_SIR_SI(ODEModel):
    """
    An age-stratified SIR model for humans, an unstratified SI model for a disease vector (f.i. mosquito)
    """

    states = ['S', 'I', 'R', 'S_v', 'I_v']
    parameters = ['beta', 'gamma']
    stratified_parameters = ['alpha']
    dimensions = ['age_group']
    state_dimensions = [['age_group'],['age_group'],['age_group'],[],[]]

    @staticmethod
    def integrate(t, S, I, R, S_v, I_v, alpha, beta, gamma):

        # Calculate total mosquito population
        N = S + I + R
        N_v = S_v + I_v
        # Calculate human differentials
        dS = -alpha*(I_v/N_v)*S
        dI = alpha*(I_v/N_v)*S - 1/gamma*I
        dR = 1/gamma*I
        # Calculate mosquito differentials
        dS_v = -np.sum(alpha*(I/N)*S_v) + (1/beta)*N_v - (1/beta)*S_v
        dI_v = np.sum(alpha*(I/N)*S_v) - (1/beta)*I_v

        return dS, dI, dR, dS_v, dI_v