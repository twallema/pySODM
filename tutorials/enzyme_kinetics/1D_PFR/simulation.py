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

#################
## Setup model ##
#################

# Design variables
# ~~~~~~~~~~~~~~~~

l = 1 # Reactor length (m)
Q = (0.200/60)*10**-6   # Flow rate (m³/s)
dt = 0.0024 # Tube inner diameter (m)
dp = 0.0004755 # Catalyst particle diameter (m)
mu = 3.35e-03 # Solvent dynamic viscosity (Pa.s)
rho_B = 545 # Catalyst density (kg/m³)
rho_F = 775 # Solvent density (kg/m³)

# Derived variables
# ~~~~~~~~~~~~~~~~~

epsilon = 0.39 + 1.74/(dt/dp+1.140)**2 # Packed bed porosity (-)
A = (dt/2)**2*math.pi # Reactor cross-sectional area (m2)
U = Q/A
u = U/epsilon # Fluid velocity (m/s)
Re = dp*U*rho_F/mu    # Reynolds number (-)

# Molar volumes
L = 1;
VaS = 0.32*L*(L-1)+9.74+4*21.95+12.79+31.57; #cm³/mol
L = 12;
VaA = 0.32*L*(L-1)+27.38+10*19.02+37.71; #cm³/mol
L = 12+1;
VaEs = 0.32*L*(L-1)+27.38+10*19.02+29.29+19.02+12.79+4*21.95+9.74; #cm³/mol
# Diffusion constants in t-Butanol
nB = 3.35e-03
T = 273.15+50
psiB = 1
MWB = 74.12
VA = np.array([VaS, VaA, VaEs, 18.75]) # molar volume (cm³/mol)
VB = 0.32*3*(3-1)+3*27.38+18.01 
D = (1e-4*(8.52*10**-8*T)/(nB*VB**(1/3))*(1.40*(VB/VA)**(1/3)+VB/VA))       
# Compute axial dispersion and kL*a
a = 6*(1-epsilon)/dp # Catalyst surface area (m-1)
kL = (0.7*D + (dp*U)/(0.18+0.008*Re**0.59))
D_ax = ((1.09/100)*(D/dp)**(2/3)*(U)**(1/3))

# Simulation parameters
N = 50 # spatial nodes

# Define reactor parameters
params={'epsilon': epsilon, 'kL_a': kL*a, 'D_ax': D_ax, 'delta_x': l/N, 'u': u, 'rho_B': rho_B}
# Append intrinsic enzyme kinetics
params.update({'Vf_Ks': 1.03/1000, 'R_AS': 1.90, 'R_AW': 2.58, 
               'R_Es': 0.57, 'K_eq': 0.89, 'K_W': 1e6, 'K_iEs':1e6})

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

fig,ax=plt.subplots()
ax.plot(coordinates['x'], out['S'].sel(phase='liquid').isel(time=-1), color='black')
ax.plot(coordinates['x'], out['Es'].sel(phase='liquid').isel(time=-1), color='black', linestyle='--')
plt.show()
plt.close()

fig,ax=plt.subplots(nrows=4,ncols=1)
for t in [0, 2000]:
    ax[0].plot(coordinates['x'], out['S'].sel(phase='liquid').isel(time=t), color='black')
    ax[0].plot(coordinates['x'], out['S'].sel(phase='solid').isel(time=t), color='black', linestyle='--')
    ax[1].plot(coordinates['x'], out['A'].sel(phase='liquid').isel(time=t), color='black')
    ax[1].plot(coordinates['x'], out['A'].sel(phase='solid').isel(time=t), color='black', linestyle='--')
    ax[2].plot(coordinates['x'], out['Es'].sel(phase='liquid').isel(time=t), color='black')
    ax[2].plot(coordinates['x'], out['Es'].sel(phase='solid').isel(time=t), color='black', linestyle='--')
    ax[3].plot(coordinates['x'], out['W'].sel(phase='liquid').isel(time=t), color='black')
    ax[3].plot(coordinates['x'], out['W'].sel(phase='solid').isel(time=t), color='black', linestyle='--')
plt.show()
plt.close()