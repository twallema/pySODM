"""
This script contains a calibration of a ping-pong bi-bi model to describe the enzymatic esterification reaction of D-Glucose and Lauric acid
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2022 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."


############################
## Load required packages ##
############################

import sys,os
import random
import emcee
import pandas as pd
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
# pySODM packages
from pySODM.optimization import pso, nelder_mead
from pySODM.optimization.utils import add_poisson_noise, assign_theta
from pySODM.optimization.mcmc import perturbate_theta, run_EnsembleSampler, emcee_sampler_to_dictionary
from pySODM.optimization.objective_functions import log_posterior_probability, log_prior_uniform, ll_gaussian
# pySODM dependecies
import corner

###############
## Load data ##
###############

experiments=[]
names = os.listdir(os.path.join(os.path.dirname(__file__),'data/'))
for name in names:
    experiments.append(pd.read_csv(os.path.join(os.path.dirname(__file__),'data/'+name), index_col=0))

################
## Load model ##
################

from models import PPBB_model

#################
## Setup model ##
#################

# Define model parameters
params={'c_enzyme': 10, 'Vf_Ks': 1.03/1000, 'R_AS': 1.90, 'R_AW': 2.58, # Forward
        'R_Es': 0.57, 'K_eq': 0.89, 'K_W': 1e6, 'K_iEs':1e6}            # Backward
# Define initial condition
init_states = {'S': 46, 'A': 61, 'W': 37, 'Es': 0}
# Initialize model
model = PPBB_model(init_states,params)

#####################
## Calibrate model ##
#####################

if __name__ == '__main__':

    #####################
    ## PSO/Nelder-Mead ##
    #####################

    # Variables
    processes = int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count()/2))
    n_pso = 30
    multiplier_pso = 20
    # Define dataset
    model.initial_states.update({'S': [experiments[-1].loc[0]['S'],], 'A': [experiments[-1].loc[0]['A'],],
                                 'W': [experiments[-1].loc[0]['W'],], 'Es': [experiments[-1].loc[0]['Es'],]})

    data=[experiments[-1]['Es'], ]
    states = ['Es',]
    weights = np.array([1,])
    log_likelihood_fnc = [ll_gaussian,]
    log_likelihood_fnc_args = [experiments[-1]['sigma'].values,]
    # Calibated parameters and bounds
    pars = ['R_Es', 'K_eq', 'K_W', 'K_iEs']
    labels = ['$R_{Es}$', '$K_{eq}$', '$K_W$', '$K_{i,Es}$']
    bounds = [(1e-6,1e6), (1e-6,10), (1e-6,1e6), (1e-6,1e6)]
    # Setup objective function (no priors --> uniform priors based on bounds)
    objective_function = log_posterior_probability(model,pars,bounds,data,states,log_likelihood_fnc,log_likelihood_fnc_args,weights,labels=labels)
    # Extract expanded bounds and labels
    expanded_labels = objective_function.expanded_labels 
    expanded_bounds = objective_function.expanded_bounds                                   
    # PSO
    #theta = pso.optimize(objective_function, kwargs={'simulation_kwargs':{'warmup': warmup}},
    #                   swarmsize=multiplier_pso*processes, maxiter=n_pso, processes=processes, debug=True)[0]    
    # Nelder-mead
    theta = [0.57, 0.89, 1e4, 1e4]
    step = len(expanded_bounds)*[0.05,]
    theta = nelder_mead.optimize(objective_function, np.array(theta), step, processes=processes, max_iter=n_pso)[0]

    ######################
    ## Visualize result ##
    ######################

    # Assign results to model
    model.parameters = assign_theta(model.parameters, pars, theta)
    # Simulate model
    out = model.sim(1500)
    # Visualize
    fig,ax=plt.subplots(figsize=(12,4))
    ax.plot(out['time'], out['S'], color='black', label='D-glucose')
    ax.plot(out['time'], out['A'], color='black', linestyle='--', label='Lauric')
    ax.plot(out['time'], out['Es'], color='red', label='Glucose laurate')
    ax.plot(out['time'], out['W'], color='red', linestyle='--', label='Water')
    ax.scatter(data[0].index, data[0].values)
    ax.legend()
    ax.grid(False)
    plt.show()
    plt.close()
    plt.tight_layout()
    plt.show()
    plt.close()

