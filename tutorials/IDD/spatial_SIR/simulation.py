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
init_states = {'S': np.array([[100, 400], [1000, 4000]]),   
               'I': np.array([[0.2, 0.8], [0, 0]])
               }
params = {'beta': 0.03,                                 # infectivity (-)
          'gamma': 5,                                   # duration of infection (d)
          'f_v': 0.1,                                   # fraction of total contacts on visited patch
          'N': np.array([[10, 10],[10, 10]]),           # contact matrix
          'M': np.array([[0.8, 0.2], [0, 1]])           # origin-destination mobility matrix
          }

# OR
# Exactly the same model without age groups

coordinates = {'age': ['0+'],  
               'location': ['A', 'B']
                }
init_states = {'S': np.array([[500, 5000]]),    
               'I': np.array([[1, 0]])
               }
params = {'beta': 0.03,                                # infectivity (-)
          'gamma': 5,                                   # duration of infection (d)
          'f_v': 0.1,                                   # fraction of total contacts on visited patch
          'N': np.mean(np.sum(params['N'], axis=1)),    # contact matrix (BE, Van Hoang, 2020)
          'M': np.array([[0.8, 0.2], [0, 1]])           # origin-destination mobility matrix
          }

# initialize model
from models import spatial_ODE_SIR
model = spatial_ODE_SIR(states=init_states, parameters=params, coordinates=coordinates)

##########################
## Simulate & visualise ##
##########################

# simulate
out = model.sim(90)

# visualise
fig,ax=plt.subplots(nrows=2, figsize=(8.3,11.7/2))

ax[0].set_title('Overall')
ax[0].plot(out['time'], out['S'].sum(dim=['age', 'location']), color='green', alpha=0.8, label='S')
ax[0].plot(out['time'], out['I'].sum(dim=['age', 'location']), color='red', alpha=0.8, label='I')
ax[0].plot(out['time'], out['R'].sum(dim=['age', 'location']), color='black', alpha=0.8, label='R')
ax[0].legend(loc=1, framealpha=1)

ax[1].set_title('Infected')
ax[1].plot(out['time'], out['I'].sum(dim='age').sel({'location': 'A'}), linestyle = '-', color='red', alpha=0.8, label='location A')
ax[1].plot(out['time'], out['I'].sum(dim='age').sel({'location': 'B'}), linestyle = '-.', color='red', alpha=0.8, label='location B')
ax[1].legend(loc=1, framealpha=1)

plt.tight_layout()
plt.show()
plt.close()


