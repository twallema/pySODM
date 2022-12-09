"""
This script contains a calibration of a ping-pong bi-bi model to describe the enzymatic esterification reaction of D-Glucose and Lauric acid
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2022 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."


############################
## Load required packages ##
############################

import math
import numpy as np
import matplotlib.pyplot as plt

################
## Load model ##
################

from models import packed_PFR


# Design variables
# ~~~~~~~~~~~~~~~~

N = 50 # Number of spatial nodes
l = 1 # Reactor length (m)
Q = (0.200/60)*10**-6   # Flow rate (m³/s)
dt = 0.0024 # Tube inner diameter (m)

# Material properties
# ~~~~~~~~~~~~~~~~~~~

dp = 0.0004755 # Catalyst particle diameter (m)
mu = 3.35e-03 # Solvent dynamic viscosity (Pa.s)
rho_B = 545 # Catalyst density (kg/m³)
rho_F = 775 # Solvent density (kg/m³)
D_AB = [3.47e-07, 2.33e-07, 1.95e-07, 1.39e-06] # Molecular diffusion coefficients in t-Butanol (m2/s)

# Derived variables
# ~~~~~~~~~~~~~~~~~

epsilon = 0.39 + 1.74/(dt/dp+1.140)**2 # Packed bed porosity (-)
A = (dt/2)**2*math.pi # Reactor cross-sectional area (m2)
U = Q/A
u = U/epsilon # Fluid velocity (m/s)
Re = dp*U*rho_F/mu    # Reynolds number (-)
a = 6*(1-epsilon)/dp # Catalyst surface area (m-1)
kL = (0.7*D_AB + (dp*U)/(0.18+0.008*Re**0.59)) # Mass transfer coefficient through boundary layer (m/s)
D_ax = ((1.09/100)*(D_AB/dp)**(2/3)*(U)**(1/3)) # Axial dispersion coefficient (m2/s)

#################
## Setup model ##
#################

# Create a dictionary of parameters
params={'epsilon': epsilon, 'kL_a': kL*a, 'D_ax': D_ax, 'delta_x': l/N, 'u': u, 'rho_B': rho_B, # Reactor
        'Vf_Ks': 1.03/1000, 'R_AS': 1.90, 'R_AW': 2.58, 'R_Es': 0.57, 'K_eq': 0.89, 'K_W': 1e6, 'K_iEs':1e6} # Enzyme kinetics

# Define coordinates
coordinates = {'phase': ['liquid','solid'], 'x': np.linspace(start=0, stop=l, num=N)}

# Define initial condition
S = np.zeros([len(coordinates['phase']), len(coordinates['x'])])
A = np.zeros([len(coordinates['phase']), len(coordinates['x'])])
S[:,0] = 30
A[:,0] = 60
W = 18*np.ones([len(coordinates['phase']), len(coordinates['x'])])
init_states = {'S': S, 'A': A, 'W': W}

# Initialize model
model = packed_PFR(init_states, params, coordinates)

####################
## Simulate model ##
####################

out = model.sim(2500)

#######################
## Visualize results ##
#######################

fig,ax=plt.subplots()
ax.plot(coordinates['x'], out['S'].sel(phase='liquid').isel(time=-1), color='black')
ax.plot(coordinates['x'], out['Es'].sel(phase='liquid').isel(time=-1), color='black', linestyle='--')
plt.show()
plt.close()