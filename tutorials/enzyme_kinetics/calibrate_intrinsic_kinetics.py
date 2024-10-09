"""
This script contains a calibration of a ping-pong bi-bi model to describe the enzymatic esterification reaction of D-Glucose and Lauric acid
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2023 by T.W. Alleman, BIOSPACE, Ghent University. All Rights Reserved."


############################
## Load required packages ##
############################

import os
import random
import datetime
import pandas as pd
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
# pySODM packages
from pySODM.optimization import pso, nelder_mead
from pySODM.optimization.utils import add_gaussian_noise, assign_theta
from pySODM.optimization.mcmc import perturbate_theta, run_EnsembleSampler, emcee_sampler_to_dictionary
from pySODM.optimization.objective_functions import log_posterior_probability, log_prior_uniform, ll_normal
# pySODM dependecies
import corner

##############
## Settings ##
##############

n_pso = 30              # Number of PSO iterations
multiplier_pso = 36     # PSO swarm size
n_mcmc = 1000           # Number of MCMC iterations
multiplier_mcmc = 18    # Total number of Markov chains = number of parameters * multiplier_mcmc
print_n = 50           # Print diagnostics every print_n iterations
discard = 100           # Discard first `discard` iterations as burn-in
thin = 10               # Thinning factor emcee chains
n = 1000                # Repeated simulations used in visualisations
processes = int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count()/2))

################
## Load model ##
################

from models import PPBB_model

#################
## Setup model ##
#################

# Define model parameters
params={'c_enzyme': 10, 'Vf_Ks': 0.95/1000, 'R_AS': 0.75, 'R_AW': 1.40, # Forward rate parameters
        'R_Es': 5.00, 'K_eq': 0.70}                                     # Backward rate parameters
# Define an initial condition
init_states = {'S': 46, 'A': 61, 'W': 37, 'Es': 0}
# Initialize model
model = PPBB_model(initial_states=init_states, parameters=params)

###############
## Load data ##
###############

# Extract and sort the names
names = os.listdir(os.path.join(os.path.dirname(__file__),'data/intrinsic_kinetics/'))
names.sort()
# Load data and prepare log likelihood function
data = []
states = []
log_likelihood_fnc = []
log_likelihood_fnc_args = []
y_err = []
initial_states=[]
for name in names:
    df = pd.read_csv(os.path.join(os.path.dirname(__file__),'data/intrinsic_kinetics/'+name), index_col=0)
    data.append(df['Es'][1:]) # Cut out zero's!
    log_likelihood_fnc.append(ll_normal)
    log_likelihood_fnc_args.append(0.04*df['Es'][1:]) # 4% Relative noise
    y_err.append(df['sigma'][1:])
    states.append('Es')
    initial_states.append(
        {'S': df.loc[0]['S'], 'A': df.loc[0]['A'], 'Es': df.loc[0]['Es'], 'W': df.loc[0]['W']}
    )

#####################
## Calibrate model ##
#####################

if __name__ == '__main__':

    #####################
    ## PSO/Nelder-Mead ##
    #####################

    # Calibated parameters and bounds
    pars = ['Vf_Ks', 'R_AS', 'R_AW', 'R_Es', 'K_eq']
    labels = ['$V_f/K_S$','$R_{AS}$','$R_{AW}$','$R_{Es}$', '$K_{eq}$']
    bounds = [(1e-5,1e-2), (1e-2,10e4), (1e-2,10e4), (1e-2,10e4), (1e-2,2)]
    # Setup objective function (no priors --> uniform priors based on bounds)
    objective_function = log_posterior_probability(model,pars,bounds,data,states,log_likelihood_fnc,log_likelihood_fnc_args,initial_states=initial_states,labels=labels)                               
    # Compute the number of cores and divide by two
    processes = int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count()/2))
    # PSO
    #theta = pso.optimize(objective_function, swarmsize=multiplier_pso*processes, max_iter=n_pso, processes=processes, debug=True)[0]    
    # Nelder-mead
    #step = len(theta)*[0.05,]
    #theta = nelder_mead.optimize(objective_function, theta, step, processes=processes, max_iter=n_pso)[0]
    theta = [0.95/1000, 0.75, 1.40, 5.00, 0.70]

    ##########
    ## MCMC ##
    ##########

    # Variables
    samples_path='sampler_output/'
    fig_path='sampler_output/'
    identifier = 'username'
    run_date = str(datetime.date.today())
    # Perturbate previously obtained estimate
    ndim, nwalkers, pos = perturbate_theta(theta, pert=len(pars)*[0.05,], multiplier=multiplier_mcmc, bounds=bounds)
    # Write some usefull settings to a .json
    settings={'start_calibration': 0, 'end_calibration': 3000,
              'n_chains': nwalkers, 'starting_estimate': list(theta), 'labels': labels}
    # Sample n_mcmc iterations
    sampler = run_EnsembleSampler(pos, n_mcmc, identifier, objective_function,
                                    fig_path=fig_path, samples_path=samples_path, print_n=print_n, backend=None, processes=processes, progress=True,
                                    settings_dict=settings)
    # Generate a sample dictionary and save it as .json for long-term storage
    samples_dict = emcee_sampler_to_dictionary(samples_path, identifier, discard=discard)
    # Look at the resulting distributions in a cornerplot
    CORNER_KWARGS = dict(smooth=0.90,title_fmt=".2E")
    fig = corner.corner(sampler.get_chain(discard=discard, thin=thin, flat=True), labels=labels, **CORNER_KWARGS)
    for idx,ax in enumerate(fig.get_axes()):
        ax.grid(False)
    plt.show()
    plt.close()

    ######################
    ## Visualize result ##
    ######################

    def draw_fcn(parameters, samples):
        idx, parameters['Vf_Ks'] = random.choice(list(enumerate(samples['Vf_Ks'])))
        parameters['R_AS'] = samples['R_AS'][idx]
        parameters['R_AW'] = samples['R_AW'][idx]
        parameters['R_Es'] = samples['R_Es'][idx]
        parameters['K_eq'] = samples['K_eq'][idx]
        return parameters
    
    # Loop over datasets
    for i,df in enumerate(data):
        # Update initial condition
        model.initial_states.update(initial_states[i])
        # Simulate model
        out = model.sim(1600, N=n, draw_function=draw_fcn, draw_function_kwargs={'samples': samples_dict})
        # Add 4% observational noise
        out = add_gaussian_noise(out, 0.04, relative=True)
        # Visualize
        fig,ax=plt.subplots(figsize=(6,2.5))
        ax.errorbar(df.index, df, yerr=2*np.squeeze(y_err[i]), capsize=10, color='black', linestyle='', marker='^', label='Data')
        ax.plot(out['time'], out['Es'].mean(dim='draws'), color='black', linestyle='--', label='Model mean')
        ax.fill_between(out['time'], out['Es'].quantile(dim='draws', q=0.025), out['Es'].quantile(dim='draws', q=0.975), color='black', alpha=0.10, label='Model 95% CI')
        ax.legend()
        ax.grid(False)
        ax.set_ylabel('Glucose Laurate (mM)')
        ax.set_xlabel('time (min)')
        plt.tight_layout()
        plt.show()
        plt.close()
