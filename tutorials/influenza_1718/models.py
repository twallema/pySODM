import pandas as pd
import numpy as np
from functools import lru_cache
from pySODM.models.base import ODEModel, SDEModel

class ODE_influenza_model(ODEModel):
    """
    Simple SEIR model for influenza with undetected carriers
    """
    
    state_names = ['S','E','Ia','Im','R','Im_inc']
    parameter_names = ['beta','sigma','gamma', 'N']
    parameter_stratified_names = ['f_a']
    dimension_names = ['age_group']

    @staticmethod
    def integrate(t, S, E, Ia, Im, R, Im_inc, beta, sigma, gamma, N, f_a):
        
        # Calculate total population
        T = S+E+Ia+Im+R
        # Calculate differentials
        dS = -beta*N@((Ia+Im)*S/T)
        dE = beta*N@((Ia+Im)*S/T) - 1/sigma*E
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
    parameter_names = ['beta','sigma','gamma','N']
    parameter_stratified_names = ['f_a']
    dimension_names = ['age_group']

    @staticmethod
    def compute_rates(t, S, E, Ia, Im, R, Im_inc, beta, sigma, gamma, N, f_a):
        
        # Protect model against 'faulty' values of f_a
        f_a = np.where(f_a < 0, 0, f_a)
        f_a = np.where(f_a > 1, 1, f_a)

        # Calculate total population
        T = S+E+Ia+Im+R

        # Compute rates per model state
        rates = {
            'S': [beta*np.matmul(N, (Ia+Im)/T),],
            'E': [f_a*(1/sigma), (1-f_a)*(1/sigma)],
            'Ia': [(1/gamma)*np.ones(T.shape),],
            'Im': [(1/gamma)*np.ones(T.shape),],
        }
        
        return rates

    @staticmethod
    def apply_transitionings(t, tau, transitionings, S, E, Ia, Im, R, Im_inc, beta, sigma, gamma, N, f_a):

        S_new  = S - transitionings['S'][0]
        E_new = E + transitionings['S'][0] - transitionings['E'][0] - transitionings['E'][1]
        Ia_new = Ia + transitionings['E'][0] - transitionings['Ia'][0]
        Im_new = Im + transitionings['E'][1] - transitionings['Im'][0]
        R_new = R + transitionings['Ia'][0] + transitionings['Im'][0]
        Im_inc_new = transitionings['E'][1]

        return S_new, E_new, Ia_new, Im_new, R_new, Im_inc_new


class make_contact_matrix_function():

    # Initialize class with contact matrices
    def __init__(self, N, N_holiday):
        self.N = N
        self.N_holiday = N_holiday

    # Define a call function to return the right contact matrix
    @lru_cache()
    def __call__(self, t, holiday=False):
        if not holiday:
            return self.N
        else:
            return self.N_holiday

    # Define a pySODM compatible wrapper with the social policies
    def contact_function(self, t, states, param, ramp_time):

        delay = pd.Timedelta(days=ramp_time)

        if t <= pd.Timestamp('2017-12-20')+delay:
            return self.__call__(t)
        # Christmass holiday
        elif pd.Timestamp('2017-12-20')+delay < t <= pd.Timestamp('2018-01-05')+delay:
            return self.__call__(t, holiday=True).copy()
        # Christmass holiday --> Winter holiday
        elif pd.Timestamp('2018-01-05')+delay < t <= pd.Timestamp('2018-02-10')+delay:
            return self.__call__(t)
        # Winter holiday
        elif pd.Timestamp('2018-02-10')+delay < t <= pd.Timestamp('2018-02-18')+delay:
            return self.__call__(t, holiday=True)
        # Winter holiday --> Easter holiday
        elif pd.Timestamp('2018-02-18')+delay < t <= pd.Timestamp('2018-03-28')+delay:
            return self.__call__(t)
        # Easter holiday
        elif pd.Timestamp('2018-03-28')+delay < t <= pd.Timestamp('2018-04-16')+delay:
            return self.__call__(t, holiday=True)
        else:
            return self.__call__(t)