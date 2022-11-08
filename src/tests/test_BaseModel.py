import pytest
import numpy as np
from ..models.base import BaseModel

# Define a model
class SIR(BaseModel):

    # state variables and parameters
    state_names = ['S', 'I', 'R']
    parameter_names = ['beta', 'gamma']

    @staticmethod
    def integrate(t, S, I, R, beta, gamma):
        """Basic SIR model"""
        N = S + I + R
        dS = -beta*I*S/N
        dI = beta*I*S/N - gamma*I
        dR = gamma*I

        return dS, dI, dR

# Setup model
parameters = {"beta": 0.9, "gamma": 0.2}
initial_states = {"S": [1_000_000 - 10], "I": [10], "R": [0]}
model = SIR(initial_states, parameters)