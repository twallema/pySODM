import numpy as np
from pySODM.models.base import ODEModel

class ODE_influenza_model(ODEModel):
    """
    Simple SEIR model for influenza with undetected carriers
    """
    
    state_names = ['S','E','Ia','Im','R','Im_inc']
    parameter_names = ['beta','sigma','gamma', 'Nc']
    parameters_stratified_names = ['f_a']
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

class SDE_influenza_model(ODEModel):
    """
    Simple stochastic SEIR model for influenza with undetected carriers
    """
    
    state_names = ['S','E','Ia','Im','R','Im_inc']
    parameter_names = ['beta','sigma','gamma','Nc']
    parameters_stratified_names = ['f_a']
    stratification_names = ['age_group']

    @staticmethod
    def integrate(t, l, S, E, Ia, Im, R, Im_inc, beta, sigma, gamma, Nc, f_a):
        
        # Calculate total population
        T = S+E+Ia+Im+R

        # Define the rates of the transitionings
        states = [S, E, Ia, Im]
        rates = [beta*np.matmul(Nc, (Ia+Im)/T), 1/sigma, 1/gamma, 1/gamma]

        # Draw the number of transitionings
        N=[]
        for i, rate in enumerate(rates):
            N.append(np.random.binomial(states[i].astype('int64'), 1-np.exp(-l*rate), size=len(S))) 

        # Update the system
        S_new  = S - N[0]
        E_new = E + N[0] - np.rint(f_a*N[1]) - np.rint((1-f_a)*N[1])
        Ia_new = Ia + np.rint(f_a*N[1]) - N[2]
        Im_new = Im + np.rint((1-f_a)*N[1]) - N[3]
        R_new = R + N[2] + N[3]
        # Calculate incidence mild disease
        Im_inc_new = np.rint((1-f_a)*N[1])/l

        return S_new, E_new, Ia_new, Im_new, R_new, Im_inc_new