"""
This script contains simulations of the enzymatic esterification reaction of D-Glucose and Lauric acid in a continuous flow reactor
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2023 by T.W. Alleman, BIOSPACE, Ghent University. All Rights Reserved."

############################
## Load required packages ##
############################

import os
import json
import math
import random
import pandas as pd
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from pySODM.optimization.utils import add_gaussian_noise

##############
## Settings ##
##############

processes = int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count()/2))
n = processes # Number of repeated simulations
end_sim = 8000 # End of simulation (s)
nx = 25 # Number of spatial nodes


######################
## Load the samples ##
######################

# Load samples
f = open(os.path.join(os.path.dirname(__file__),'data/username_SAMPLES_2023-06-07.json'))
samples_dict = json.load(f)

# Define draw function
def draw_fcn(parameters, samples):
    idx, parameters['Vf_Ks'] = random.choice(list(enumerate(samples['Vf_Ks'])))
    parameters['R_AS'] = samples['R_AS'][idx]
    parameters['R_AW'] = samples['R_AW'][idx]
    parameters['R_Es'] = samples['R_Es'][idx]
    parameters['K_eq'] = samples['K_eq'][idx]
    return parameters

###################
## Load the data ##
###################

reactor_cutting = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data/1D_PFR/reactor_cutting.csv'), index_col=0)
vary_flowrate = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data/1D_PFR/vary_flowrate.csv'), index_col=0)

######################
## Model parameters ##
######################

# Design variables
# ~~~~~~~~~~~~~~~~

l = 1 # Reactor length (m)
Q = (0.20/60)*10**-6   # Flow rate (m³/s)
dt = 0.0024 # Tube inner diameter (m)

# Material properties
# ~~~~~~~~~~~~~~~~~~~

dp = 0.0004755 # Catalyst particle diameter (m)
mu = 3.35e-03 # Solvent dynamic viscosity (Pa.s)
rho_B = 545 # Catalyst density (kg/m³)
rho_F = 775 # Solvent density (kg/m³)
D_AB = np.array([3.47e-07, 2.33e-07, 1.95e-07, 1.39e-06]) # Molecular diffusion coefficients in t-Butanol (m2/s)

# Derived variables
# ~~~~~~~~~~~~~~~~~

epsilon = 0.39 + 1.74/(dt/dp+1.140)**2 # Packed bed porosity (-)
A = (dt/2)**2*math.pi # Reactor cross-sectional area (m2)
U = Q/A
u = U/epsilon # Fluid velocity (m/s)
Re = dp*U*rho_F/mu/(1-epsilon)    # Reynolds number (-)
a = 6*(1-epsilon)/dp # Catalyst surface area (m-1)
kL = (0.7*D_AB + (dp*U)/(0.18+0.008*Re**0.59)) # Mass transfer coefficient through boundary layer (m/s)
D_ax = ((1.09/100)*(D_AB/dp)**(2/3)*(U)**(1/3)) # Axial dispersion coefficient (m2/s)

################
## Load model ##
################

from models import packed_PFR

#################
## Setup model ##
#################

# Create a dictionary of parameters
params={'epsilon': epsilon, 'kL_a': kL*a, 'D_ax': D_ax, 'delta_x': l/nx, 'u': u, 'rho_B': rho_B, # Reactor
        'Vf_Ks': 7.88e-04, 'R_AS': 0.055, 'R_AW': 2.64, 'R_Es': 0.079, 'K_eq': 0.418} # Enzyme kinetics

# Define coordinates
coordinates = {'species': ['S','A','Es','W'], 'x': np.linspace(start=0, stop=l, num=nx)}

# Define initial concentrations
initial_concentrations = [30,60,0,28]

# Initialise initial states
C_F = np.zeros([len(coordinates['species']), len(coordinates['x'])])
C_S = np.zeros([len(coordinates['species']), len(coordinates['x'])])
# Initialize inlet concentrations
C_F[:,0] = initial_concentrations
C_S[:,0] = initial_concentrations
# t-Butanol with water has already equilibrated inside the reactor
C_F[3,:] = initial_concentrations[3]
C_S[3,:] = initial_concentrations[3]

# Initialize model
model = packed_PFR({'C_F': C_F, 'C_S': C_S}, params, coordinates)

###########################
## Concentration profile ##
###########################

out = model.sim(end_sim, N=n, draw_function=draw_fcn, draw_function_kwargs={'samples': samples_dict}, processes=processes)
# Add 4% observational noise
out = add_gaussian_noise(out, 0.04, relative=True)
# Visualize 
fig,ax=plt.subplots(figsize=(6,2.5))
# Data
y_error = [2*(reactor_cutting['mean'] - reactor_cutting['lower']), 2*(reactor_cutting['upper'] - reactor_cutting['mean'])]
ax.errorbar(reactor_cutting.index, reactor_cutting['mean'], yerr=y_error, capsize=10,
            color='black', linestyle='', marker='^', label='Data 95% CI')
# Model prediction
ax.plot(coordinates['x'], out['C_F'].sel(species='Es').isel(time=-1).mean(dim='draws'), color='black', linestyle='--', label='Model mean')
ax.fill_between(coordinates['x'], out['C_F'].sel(species='Es').isel(time=-1).quantile(dim='draws', q=0.025), out['C_F'].sel(species='Es').isel(time=-1).quantile(dim='draws', q=0.975), color='black', alpha=0.10, label='Model 95% CI')

ax.set_xlabel('Reactor length (m)')
ax.set_ylabel('Ester concentration (mM)')
ax.legend()
plt.tight_layout()
plt.show()
plt.close()

#######################
## Vary the flowrate ##
#######################

# Different initial condition

# Define initial concentrations
initial_concentrations = [30,60,0,18]
# Initialise initial states
C_F = np.zeros([len(coordinates['species']), len(coordinates['x'])])
C_S = np.zeros([len(coordinates['species']), len(coordinates['x'])])
# Initialize inlet concentrations
C_F[:,0] = initial_concentrations
C_S[:,0] = initial_concentrations
# t-Butanol with water has already equilibrated inside the reactor
C_F[3,:] = initial_concentrations[3]
C_S[3,:] = initial_concentrations[3]

# Different length
l = 0.6 # m

# Loop over flowrates
out=[]
Q = np.linspace(start=0.05, stop=0.6, num=50)/60*10**-6  # Flow rate (m³/s)
for q in Q:
    print(f'Computing flowrate: {q} m3/s')
    # Update derived variables
    U = q/A
    u = U/epsilon # Fluid velocity (m/s)
    Re = dp*U*rho_F/mu    # Reynolds number (-)
    a = 6*(1-epsilon)/dp # Catalyst surface area (m-1)
    kL = (0.7*D_AB + (dp*U)/(0.18+0.008*Re**0.59)) # Mass transfer coefficient through boundary layer (m/s)
    D_ax = ((1.09/100)*(D_AB/dp)**(2/3)*(U)**(1/3)) # Axial dispersion coefficient (m2/s)
    # Update model parameters
    model.parameters.update({'kL_a': kL*a, 'D_ax': D_ax, 'delta_x': l/nx, 'u': u}) 
    # Simulate
    out_tmp = model.sim(end_sim, N=n, draw_function=draw_fcn, draw_function_kwargs={'samples': samples_dict}, processes=processes)
    # Add 4% observational noise and store
    out.append(add_gaussian_noise(out_tmp, 0.04, relative=True))

# Compute outlet concentrations
mean_outlet=[]
lower_outlet=[]
upper_outlet=[]
for output in out:
    mean_outlet.append(float(output['C_F'].sel(species='Es').isel(time=-1).isel(x=-1).mean(dim='draws').values))
    lower_outlet.append(float(output['C_F'].sel(species='Es').isel(time=-1).isel(x=-1).quantile(dim='draws', q=0.025).values))
    upper_outlet.append(float(output['C_F'].sel(species='Es').isel(time=-1).isel(x=-1).quantile(dim='draws', q=0.975).values))

# Visualize results
fig,ax=plt.subplots(figsize=(6,2.6))
# Data
y_error = [2*(vary_flowrate['mean'] - vary_flowrate['lower']), 2*(vary_flowrate['upper'] - vary_flowrate['mean'])]
ax.errorbar(vary_flowrate.index, vary_flowrate['mean'], yerr=y_error, capsize=5,color='black', linestyle='', marker='^', label='Data 95% CI')
# Model prediction: outlet concentration
ax.plot(Q*60*10**6, mean_outlet, color='black', linestyle='--', label='Model mean')
ax.fill_between(Q*60*10**6, lower_outlet, upper_outlet, color='black', alpha=0.10, label='Model 95% CI' )
# Decorations
ax.set_xlabel('Flow rate (mL/min)')
ax.set_ylabel('Outlet ester concentration (mM)')
ax.set_ylim([-0.5,20])
ax.legend()
plt.tight_layout()
plt.show()
plt.close()