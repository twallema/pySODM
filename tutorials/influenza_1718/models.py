import pandas as pd
import numpy as np
from pySODM.models.base import ODE, JumpProcess

class ODE_influenza_model(ODE):
    """
    Simple SEIR model for influenza with undetected carriers
    """
    
    states = ['S','E','Ip','Iud','Id','R','Im_inc']
    parametes = ['alpha', 'beta', 'gamma', 'delta','N']
    stratified_parameters = ['f_ud']
    dimensions = ['age_group']

    @staticmethod
    def integrate(t, S, E, Ip, Iud, Id, R, Im_inc, alpha, beta, gamma, delta, N, f_ud):
        
        # Calculate total population
        T = S+E+Ip+Iud+Id+R
        # Calculate differentials
        dS = -beta*N@((Ip+Iud+0.22*Id)*S/T)
        dE = beta*N@((Ip+Iud+0.22*Id)*S/T) - 1/alpha*E
        dIp = 1/alpha*E - 1/gamma*Ip
        dIud = f_ud/gamma*Ip - 1/delta*Iud
        dId = (1-f_ud)/gamma*Ip - 1/delta*Id
        dR = 1/delta*(Iud+Id)
        # Calculate incidence mild disease
        dIm_inc_new = (1-f_ud)/gamma*Ip - Im_inc

        return dS, dE, dIp, dIud, dId, dR, dIm_inc_new

class JumpProcess_influenza_model(JumpProcess):
    """
    Simple stochastic SEIR model for influenza with undetected carriers
    """
    
    states = ['S','E','Ip','Iud','Id','R','Im_inc']
    parameters = ['alpha', 'beta', 'gamma', 'delta','N']
    stratified_parameters = ['f_ud']
    dimensions = ['age_group']

    @staticmethod
    def compute_rates(t, S, E, Ip, Iud, Id, R, Im_inc, alpha, beta, gamma, delta, N, f_ud):
        
        # Calculate total population
        T = S+E+Ip+Iud+Id+R
        # Compute rates per model state
        rates = {
            'S': [beta*np.matmul(N, (Ip+Iud+0.22*Id)/T),],
            'E': [1/alpha*np.ones(T.shape),],
            'Ip': [f_ud*(1/gamma), (1-f_ud)*(1/gamma)],
            'Iud': [(1/delta)*np.ones(T.shape),],
            'Id': [(1/delta)*np.ones(T.shape),],
        }
        
        return rates

    @staticmethod
    def apply_transitionings(t, tau, transitionings, S, E, Ip, Iud, Id, R, Im_inc, alpha, beta, gamma, delta, N, f_ud):

        S_new  = S - transitionings['S'][0]
        E_new = E + transitionings['S'][0] - transitionings['E'][0]
        Ip_new = Ip + transitionings['E'][0] - transitionings['Ip'][0] - transitionings['Ip'][1]
        Iud_new = Iud + transitionings['Ip'][0] - transitionings['Iud'][0]
        Id_new = Id + transitionings['Ip'][1] - transitionings['Id'][0]
        R_new = R + transitionings['Iud'][0] + transitionings['Id'][0]
        Im_inc_new = transitionings['Ip'][1]

        return S_new, E_new, Ip_new, Iud_new, Id_new, R_new, Im_inc_new


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

        if t <= pd.Timestamp('2017-10-30'):
            return self.__call__(t)
        # Winter holiday
        elif pd.Timestamp('2017-10-30') < t <= pd.Timestamp('2017-11-05'):
            return self.__call__(t, holiday=True)
        # Winter holiday --> Christmas holiday
        elif pd.Timestamp('2017-11-05') < t <= pd.Timestamp('2017-12-22'):
            return self.__call__(t)    
        # Christmas holiday
        elif pd.Timestamp('2017-12-22') < t <= pd.Timestamp('2018-01-07'):
            return self.__call__(t, holiday=True)
        # Christmas holiday --> Spring holiday
        elif pd.Timestamp('2018-01-07') < t <= pd.Timestamp('2018-02-12'):
            return self.__call__(t)
        #Spring holiday
        elif pd.Timestamp('2018-02-12') < t <= pd.Timestamp('2018-02-18'):
            return self.__call__(t, holiday=True)
        # Spring holiday --> Easter holiday
        elif pd.Timestamp('2018-02-18') < t <= pd.Timestamp('2018-04-02'):
            return self.__call__(t)
        # Easter holiday
        elif pd.Timestamp('2018-04-02') < t <= pd.Timestamp('2018-04-15'):
            return self.__call__(t, holiday=True)
        else:
            return self.__call__(t)