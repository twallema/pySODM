from pySODM.models.base import ODEModel

class PPBB_model(ODEModel):
    """
    A model for the enzymatic esterification conversion of D-Glucose and Lauric acid into Glucose Laurate Ester and water
    S + A <--> Es + W
    """
    
    state_names = ['S','A','Es','W']
    parameter_names = ['c_enzyme', 'Vf_Ks', 'R_AS', 'R_AW', 'R_Es', 'K_eq', 'K_W', 'K_iEs']

    @staticmethod
    def integrate(t, S, A, Es, W, c_enzyme, Vf_Ks, R_AS, R_AW, R_Es, K_eq, K_W, K_iEs):

        # Calculate rate
        v = c_enzyme*(Vf_Ks*(S*A - (1/K_eq)*Es*W)/(A + R_AS*S + R_AW*W + R_AS*S*Es/K_iEs + R_Es*Es + R_Es*W*Es/K_W))
       
        return -v, -v, v, v