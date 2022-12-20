import numpy as np
from pySODM.models.base import ODEModel, SDEModel

class ODE_influenza_model(ODEModel):
    """
    Simple SEIR model for influenza with undetected carriers
    """
    
    state_names = ['S','E','Ia','Im','R','Im_inc']
    parameter_names = ['sigma','gamma', 'Nc']
    parameter_stratified_names = ['beta', 'f_a']
    stratification_names = ['age_group']

    @staticmethod
    def integrate(t, S, E, Ia, Im, R, Im_inc, beta, sigma, gamma, Nc, f_a):
        
        # Calculate total population
        T = S+E+Ia+Im+R
        # Calculate differentials
        dS = -beta*Nc@((Ia+Im)*S/T)
        dE = beta*Nc@((Ia+Im)*S/T) - 1/sigma*E
        dIa = f_a*E/sigma - 1/gamma*Ia
        dIm = (1-f_a)/sigma*E - 1/gamma*Im
        dR = 1/gamma*(Ia+Im)
        # Calculate incidence mild disease
        dIm_inc_new = (1-f_a)/sigma*E - Im_inc

        return dS, dE, dIa, dIm, dR, dIm_inc_new

class SDE_influenza_model(SDEModel):
    """
    Simple stochastic SEIR model for influenza with undetected carriers
    """
    
    state_names = ['S','E','Ia','Im','R','Im_inc']
    parameter_names = ['beta','sigma','gamma','Nc']
    parameter_stratified_names = ['f_a']
    stratification_names = ['age_group']

    @staticmethod
    def compute_rates(t, S, E, Ia, Im, R, Im_inc, beta, sigma, gamma, Nc, f_a):
        
        # Calculate total population
        T = S+E+Ia+Im+R

        # Rates per model state
        rates = {
            'S': [beta*np.matmul(Nc, (Ia+Im)/T),],
            'E': [f_a*(1/sigma), (1-f_a)*(1/sigma)],
            'Ia': [(1/gamma)*np.ones(T.shape),],
            'Im': [(1/gamma)*np.ones(T.shape),],
        }

        return rates

    @staticmethod
    def apply_transitionings(t, tau, transitionings, S, E, Ia, Im, R, Im_inc, beta, sigma, gamma, Nc, f_a):

        S_new  = S - transitionings['S'][0]
        E_new = E + transitionings['S'][0] - transitionings['E'][0] - transitionings['E'][1]
        Ia_new = Ia + transitionings['E'][0] - transitionings['Ia'][0]
        Im_new = Im + transitionings['E'][1] - transitionings['Im'][0]
        R_new = R + transitionings['Ia'][0] + transitionings['Im'][0]
        Im_inc_new = transitionings['E'][1]

        return S_new, E_new, Ia_new, Im_new, R_new, Im_inc_new

