import pandas as pd
import numpy as np
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

    def __init__(self, N):
        self.N = N

    def __call__(self, t, holiday=False):
        """
        A function to choose the appropriate contact matrix (holiday? weekend?)
        """

        # Choose between holiday/no_holiday
        if holiday:
            N = self.N['holiday']
        else:
            N = self.N['no_holiday']
        
        # Choose between weekday and weekendday
        if ((t.weekday() == 5) | (t.weekday() == 6)):
            return N['weekend']
        else:
            return N['week']

    # Define a pySODM compatible wrapper with the social policies
    def contact_function(self, t, states, param):
        """
        A pySODM compatible wrapper containing the social policies

        Input
        =====

        t: timestamp
            Current simulated date

        states: dict
            Dictionary containing model states at current time

        param: dict
            Dictionary containing all model parameters

        Output
        ======

        N: np.ndarray (4x4)
            Matrix of social contacts at time `t`

        """
        t = pd.to_datetime(t)

        if t <= pd.Timestamp('2017-12-22'):
            return self.__call__(t)
        # Christmas holiday
        elif pd.Timestamp('2017-12-22') < t <= pd.Timestamp('2018-01-07'):
            return self.__call__(t, holiday=True)
        # Christmas holiday --> Winter holiday
        elif pd.Timestamp('2018-01-07') < t <= pd.Timestamp('2018-02-12'):
            return self.__call__(t)
        # Winter holiday
        elif pd.Timestamp('2018-02-12') < t <= pd.Timestamp('2018-02-18'):
            return self.__call__(t, holiday=True)
        # Winter holiday --> Easter holiday
        elif pd.Timestamp('2018-02-18') < t <= pd.Timestamp('2018-04-02'):
            return self.__call__(t)
        # Easter holiday
        elif pd.Timestamp('2018-04-02') < t <= pd.Timestamp('2018-04-15'):
            return self.__call__(t, holiday=True)
        else:
            return self.__call__(t)