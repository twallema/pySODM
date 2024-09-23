"""
This script contains a calibration of an SIR model to synthetic data.
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

############################
## Load required packages ##
############################

import numpy as np
import matplotlib.pyplot as plt

######################
## Initialise model ##
######################

# A model with two age groups and two spatial patches 'A' (500 inhabitants) and 'B' (5000 inhabitants).
# Epidemic is seeded in location 'A'. Inhabitants of 'B' never visit 'A'.
# While 20% of inhabitants of 'A' visits 'B', and has f_v % of their contacts in 'B'.

coordinates = {'age': ['0-25', '25+'],  
               'location': ['A', 'B']  
                }
init_states = {'S': np.array([[100, 1000], [400, 4000]]),   
               'I': np.array([[0.2, 0], [0.8, 0]])
               }
params = {'beta': 0.03,                                 # infectivity (-)
          'gamma': 5,                                   # duration of infection (d)
          'f_v': 0.1,                                   # fraction of total contacts on visited patch
          'N': np.array([[10, 10], [10, 10]]),          # contact matrix
          'M': np.array([[0.8, 0.2], [0, 1]])           # origin-destination mobility matrix
          }

# initialize model
from models import spatial_ODE_SIR
model = spatial_ODE_SIR(states=init_states, parameters=params, coordinates=coordinates)

##########################
## Simulate & visualise ##
##########################

# simulate
out_det = model.sim(90)

# visualise
fig,ax=plt.subplots(nrows=2, figsize=(8.3,11.7/2))

ax[0].set_title('Overall')
ax[0].plot(out_det['time'], out_det['S'].sum(dim=['age', 'location']), color='green', alpha=0.8, label='S')
ax[0].plot(out_det['time'], out_det['I'].sum(dim=['age', 'location']), color='red', alpha=0.8, label='I')
ax[0].plot(out_det['time'], out_det['R'].sum(dim=['age', 'location']), color='black', alpha=0.8, label='R')
ax[0].legend(loc=1, framealpha=1)

ax[1].set_title('Infected')
ax[1].plot(out_det['time'], out_det['I'].sum(dim='age').sel({'location': 'A'}), linestyle = '-', color='red', alpha=0.8, label='location A')
ax[1].plot(out_det['time'], out_det['I'].sum(dim='age').sel({'location': 'B'}), linestyle = '-.', color='red', alpha=0.8, label='location B')
ax[1].legend(loc=1, framealpha=1)

plt.tight_layout()
plt.show()
plt.close()

import sys
sys.exit()

######################
### initialize model #
######################

from models import matmul_2D_3D_matrix, spatial_TL_SIR # kan je importeren uit ander python script
init_states['S_work'] = np.atleast_2d(matmul_2D_3D_matrix(init_states['S'], params['M'])) # vermijd code copieren --> zo ga je plots "onverklaarbare" fouten maken tegen het einde van een script
model_stoch = spatial_TL_SIR(states=init_states, parameters=params, coordinates=coordinates)

#############
## Simulate #
#############

# simulate one single time
out_stoch = model_stoch.sim(90, N=100)   # repeated simulations zitten ingebouwd in pySODM
av_TL = out_stoch.mean(dim="draws")     # onder dimensienaam 'draws'

##########################
# visualise ODE against TL
##########################
line_styles = [ '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 10))]
fig, ax = plt.subplots(2, 2, figsize=(15, 12))

ax[0,0].set_title('Overall', fontsize = 12)
ax[0,0].plot(out_det['time'], out_det['S'].sum(dim=['age', 'location']), color='green', alpha=0.8, label='S')
ax[0,0].plot(out_det['time'], out_det['I'].sum(dim=['age', 'location']), color='red', alpha=0.8, label='I')
ax[0,0].plot(out_det['time'], out_det['R'].sum(dim=['age', 'location']), color='black', alpha=0.8, label='R')
for idx, age_group in enumerate(out_det['age']):
    ax[0, 0].plot(out_det['time'], out_det['I'].sel(age=age_group).sum(dim='location'), 
                  color='red', linestyle=line_styles[idx % len(line_styles)], alpha=0.8, 
                  label=f'I (age {age_group.values})')

ax[0, 0].legend(loc=1, framealpha=1)

ax[0,1].set_title('Infected', fontsize = 12)
ax[0,1].plot(out_det['time'], out_det['I'].sum(dim='age').sel({'location': 'A'}), linestyle = '-', color='red', alpha=0.8, label='location A')
ax[0,1].plot(out_det['time'], out_det['I'].sum(dim='age').sel({'location': 'B'}), linestyle = '-.', color='red', alpha=0.8, label='location B')
ax[0,1].legend(loc=1, framealpha=1)

ax[1,0].plot(av_TL['time'], av_TL['S'].sum(dim=['age', 'location']), color='green', alpha=0.8, label='S')
ax[1,0].plot(av_TL['time'], av_TL['I'].sum(dim=['age', 'location']), color='red', alpha=0.8, label='I')
ax[1,0].plot(av_TL['time'], av_TL['R'].sum(dim=['age', 'location']), color='black', alpha=0.8, label='R')
for idx, age_group in enumerate(out_det['age']):
    ax[1, 0].plot(av_TL['time'], av_TL['I'].sel(age=age_group).sum(dim='location'), 
                  color='red', linestyle=line_styles[idx % len(line_styles)], alpha=0.8, 
                  label=f'I (age {age_group.values})')

ax[0, 0].legend(loc=1, framealpha=1)
ax[1,0].legend(loc=1, framealpha=1)

ax[1,1].plot(av_TL['time'], av_TL['I'].sum(dim='age').sel({'location': 'A'}), linestyle = '-', color='red', alpha=0.8, label='location A')
ax[1,1].plot(av_TL['time'], av_TL['I'].sum(dim='age').sel({'location': 'B'}), linestyle = '-.', color='red', alpha=0.8, label='location B')
ax[1,1].legend(loc=1, framealpha=1)

fig.text(0.5, 0.95, 'Deterministic model - with ages', ha='center', fontsize=16)
fig.text(0.5, 0.47, 'Stochastic model - with ages', ha='center', fontsize=16)

# Adjust layout to increase spacing between rows and make space for titles
plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=4.0)  # Increase h_pad to add more space between rows overarching titles
# plt.show()
# plt.close()

#############################
# NOW WITH REDUCED DIMENSIONS 
#############################

# Exactly the same model without_det age groups
# Validated the reduction to one spatial unit as well (becomes a plain SIR)

###########
# ODE MODEL  - no age-groups
###########

# initialize model
coordinates = {'age': ['0+'],  
               'location': ['A', 'B']
                }
init_states = {'S': np.array([[500, 5000]]),    
               'I': np.array([[10, 0]])
               }
params = {'beta': 0.03,                                 # infectivity (-)
          'gamma': 5,                                   # duration of infection (d)
          'f_v': 0.1,                                   # fraction of total contacts on visited patch
          'N': 20.0,                                    # contact matrix
          'M': np.array([[0.8, 0.2], [0, 1]])           # origin-destination mobility matrix
          }

from models import spatial_ODE_SIR
model = spatial_ODE_SIR(states=init_states, parameters=params, coordinates=coordinates)

# SIMULATE ODE MODEL
out_det = model.sim(90)

####################
# STOCHASTIC MODEL - no age-groups
####################

init_states['S_work'] = np.atleast_2d(matmul_2D_3D_matrix(init_states['S'], params['M'])) 
model_stoch = spatial_TL_SIR(states=init_states, parameters=params, coordinates=coordinates)

# SIMULATE MODEL 
TL_out = model_stoch.sim(90, N=100)
av_TL = TL_out.mean(dim='draws')

##########################
# visualise ODE against TL - no age groups
##########################

fig, ax = plt.subplots(2, 2, figsize=(15, 12))

ax[0,0].set_title('Overall', fontsize = 12)
ax[0,0].plot(out_det['time'], out_det['S'].sum(dim=['age', 'location']), color='green', alpha=0.8, label='S')
ax[0,0].plot(out_det['time'], out_det['I'].sum(dim=['age', 'location']), color='red', alpha=0.8, label='I')
ax[0,0].plot(out_det['time'], out_det['R'].sum(dim=['age', 'location']), color='black', alpha=0.8, label='R')
ax[0,0].legend(loc=1, framealpha=1)

ax[0,1].set_title('Infected', fontsize = 12)
ax[0,1].plot(out_det['time'], out_det['I'].sum(dim='age').sel({'location': 'A'}), linestyle = '-', color='red', alpha=0.8, label='location A')
ax[0,1].plot(out_det['time'], out_det['I'].sum(dim='age').sel({'location': 'B'}), linestyle = '-.', color='red', alpha=0.8, label='location B')
ax[0,1].legend(loc=1, framealpha=1)

ax[1,0].plot(av_TL['time'], av_TL['S'].sum(dim=['age', 'location']), color='green', alpha=0.8, label='S')
ax[1,0].plot(av_TL['time'], av_TL['I'].sum(dim=['age', 'location']), color='red', alpha=0.8, label='I')
ax[1,0].plot(av_TL['time'], av_TL['R'].sum(dim=['age', 'location']), color='black', alpha=0.8, label='R')
ax[1,0].legend(loc=1, framealpha=1)

ax[1,1].plot(av_TL['time'], av_TL['I'].sum(dim='age').sel({'location': 'A'}), linestyle = '-', color='red', alpha=0.8, label='location A')
ax[1,1].plot(av_TL['time'], av_TL['I'].sum(dim='age').sel({'location': 'B'}), linestyle = '-.', color='red', alpha=0.8, label='location B')
ax[1,1].legend(loc=1, framealpha=1)

fig.text(0.5, 0.95, 'Deterministic model - no ages', ha='center', fontsize=16)
fig.text(0.5, 0.47, 'Stochastic model - no ages', ha='center', fontsize=16)

# Adjust layout to increase spacing between rows and make space for titles
plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=4.0)  # Increase h_pad to add more space between rows overarching titles
plt.show()
# plt.close()

# difference between Stochastic and Deterministic is big when removing the age-groups, something is not functioning correctly