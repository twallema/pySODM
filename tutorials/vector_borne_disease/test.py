"""
This script contains a calibration of an influenza model to 2017-2018 data.
"""

__author__      = "Tijs Alleman & Wolf Demunyck"
__copyright__   = "Copyright (c) 2022 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."


############################
## Load required packages ##
############################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

##################
## Define model ##
##################

# Import the ODEModel class
from models import SDE_SIR_SI as SIR_SI

# Define parameters and initial condition
params={'alpha': 10*np.array([0.005, 0.01, 0.02, 0.015]), 'gamma': 5, 'beta': 7}
init_states = {'S': [606938, 1328733, 7352492, 2204478], 'S_v': 1e6, 'I_v': 2}
# Define model coordinates
age_groups = pd.IntervalIndex.from_tuples([(0,5),(5,15),(15,65),(65,120)])
coordinates={'age_group': age_groups}

# Initialize model
model = SIR_SI(states=init_states, parameters=params, coordinates=coordinates)

# Simulate model
out = model.sim(365)
print(out)

# Visualize result
fig,ax=plt.subplots(nrows=2, ncols=1, figsize=(8,6), sharex=True)
ax[0].plot(out['time'], out['S_v']/init_states['S_v']*100, color='green', label='Susceptible')
ax[0].plot(out['time'], out['I_v']/init_states['S_v']*100, color='red', label='Infected')
ax[0].set_ylabel('Health state vectors (%)')
ax[0].legend()
ax[0].set_title('Vector lifespan: ' + str(params['beta']) + ' days')
colors=['black', 'red', 'green', 'blue']
labels=['0-5','5-15','15-65','65-120']
for i,age_group in enumerate(age_groups):
    ax[1].plot(out['time'], out['I'].sel(age_group=age_group)/init_states['S'][i]*100000, color=colors[i], label=labels[i])
ax[1].set_ylabel('Infectious humans per 100K')
ax[1].legend()
ax[1].set_xlabel('time (days)')
ax[1].set_title('Vector-to-human transfer rate: '+str(params['alpha']))
plt.show()
plt.close()