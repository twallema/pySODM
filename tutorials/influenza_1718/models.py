from pySODM.models.base import BaseModel

class influenza_model(BaseModel):
    """
    Simple SEIR model for influenza with undetected carriers
    """
    
    state_names = ['S','E','Ia','Im','R','Im_new']
    parameter_names = ['beta','sigma','f_a','gamma']
    stratification = ['Nc']

    @staticmethod
    def integrate(t, S, E, Ia, Im, R, Im_new, beta, sigma, f_a, gamma, Nc):
        
        # Calculate total population
        T = S+E+Ia+Im+R
        # Calculate differentials
        dS = -beta*Nc@((Ia+Im)*S/T)
        dE = beta*Nc@((Ia+Im)*S/T) - 1/sigma*E
        dIa = f_a*E/sigma - 1/gamma*Ia
        dIm = (1-f_a)/sigma*E - 1/gamma*Im
        dR = 1/gamma*(Ia+Im)
        # Calculate incidence mild disease
        dIm_new = (1-f_a)/sigma*E - Im_new

        return dS, dE, dIa, dIm, dR, dIm_new
