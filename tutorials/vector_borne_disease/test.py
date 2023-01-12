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
from models import SIR_SI

# Define parameters and initial condition
params={'alpha': [0.01, 0.01, 0.02, 0.01], 'gamma': 5}
init_states = {'S': [606938, 1328733, 7352492, 2204478], 'S_v': 1e6, 'I_v': 1}
# Define model coordinates
age_groups = pd.IntervalIndex.from_tuples([(0,5),(5,15),(15,65),(65,120)])
coordinates={'age_group': age_groups}

# Initialize model
model = SIR_SI(states=init_states, parameters=params, coordinates=coordinates)

# Simulate model
out = model.sim(121)

# Visualize result
fig,ax=plt.subplots(figsize=(8,6))
ax.plot(out['time'], out['S_v'], color='green')
ax.plot(out['time'], out['I_v'], color='red')
for age_group in age_groups:
    ax.plot(out['time'], out['I'].sel(age_group=age_group), color='black')
plt.show()
plt.close()