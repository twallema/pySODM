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
import datetime
import emcee
import pandas as pd
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
# pySODM packages
from pySODM.optimization import pso, nelder_mead
from pySODM.optimization.utils import add_gaussian_noise, assign_theta
from pySODM.optimization.mcmc import perturbate_theta, run_EnsembleSampler, emcee_sampler_to_dictionary
from pySODM.optimization.objective_functions import log_posterior_probability, log_prior_uniform, ll_gaussian
# pySODM dependecies
import corner

# Number of repeated simulations with sampled parameters to visualize goodness-of-fit
N = 50

###############
## Load data ##
###############

# Extract and sort the names
names = os.listdir(os.path.join(os.path.dirname(__file__),'data/intrinsic_kinetics/'))
names.sort()

# Load data
data = []
states = []
log_likelihood_fnc = []
log_likelihood_fnc_args = []
initial_states=[]
for name in names:
    df = pd.read_csv(os.path.join(os.path.dirname(__file__),'data/intrinsic_kinetics/'+name), index_col=0)
    data.append(df['Es'])
    log_likelihood_fnc.append(ll_gaussian)
    log_likelihood_fnc_args.append(df['sigma'])
    states.append('Es')
    initial_states.append(
        {'S': df.loc[0]['S'], 'A': df.loc[0]['A'], 'Es': df.loc[0]['Es'], 'W': df.loc[0]['W']}
    )

################
## Load model ##
################

from models import PPBB_model

#################
## Setup model ##
#################

# Define model parameters
params={'c_enzyme': 10, 'Vf_Ks': 1.03/1000, 'R_AS': 1.90, 'R_AW': 2.58, # Forward
        'R_Es': 0.57, 'K_eq': 0.89}                                     # Backward
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

    # Calibated parameters and bounds
    pars = ['Vf_Ks', 'R_AS', 'R_AW', 'R_Es', 'K_eq']
    labels = ['$V_f/K_S$','$R_{AS}$','$R_{AW}$','$R_{Es}$', '$K_{eq}$']
    bounds = [(1e-5,1e-2), (1e-2,1000), (1e-2,1000), (1e-2,1000), (1e-2,2)]
    # Setup objective function (no priors --> uniform priors based on bounds)
    objective_function = log_posterior_probability(model,pars,bounds,data,states,log_likelihood_fnc,log_likelihood_fnc_args,initial_states=initial_states,labels=labels)                               
    # Settings
    processes = int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count()/2))
    n_pso = 50
    multiplier_pso = 20
    # PSO
    theta = pso.optimize(objective_function, swarmsize=multiplier_pso*processes, max_iter=n_pso, processes=processes, debug=True)[0]    
    # Nelder-mead
    step = len(theta)*[0.05,]
    theta = nelder_mead.optimize(objective_function, theta, step, processes=processes, max_iter=n_pso)[0]

    ##########
    ## MCMC ##
    ##########

    # Variables
    n_mcmc = 200
    print_n = 50
    discard = 100
    samples_path='sampler_output/'
    fig_path='sampler_output/'
    identifier = 'username'
    run_date = str(datetime.date.today())
    # Perturbate previously obtained estimate
    ndim, nwalkers, pos = perturbate_theta(theta, pert=[0.10, 0.10, 0.10, 0.10, 0.10], multiplier=5, bounds=bounds)
    # Write some usefull settings to a .json
    settings={'start_calibration': 0, 'end_calibration': 3000,
              'n_chains': nwalkers, 'starting_estimate': list(theta), 'labels': labels}
    # Sample 100 iterations
    sampler = run_EnsembleSampler(pos, n_mcmc, identifier, objective_function,
                                    fig_path=fig_path, samples_path=samples_path, print_n=print_n, backend=None, processes=processes, progress=True,
                                    settings_dict=settings)
    # Sample 100 more
    backend = emcee.backends.HDFBackend(os.path.join(os.getcwd(),samples_path+'username_BACKEND_'+run_date+'.hdf5'))
    sampler = run_EnsembleSampler(pos, n_mcmc, identifier, objective_function,
                                    fig_path=fig_path, samples_path=samples_path, print_n=print_n, backend=backend, processes=processes, progress=True,
                                    settings_dict=settings)
    # Generate a sample dictionary and save it as .json for long-term storage
    samples_dict = emcee_sampler_to_dictionary(samples_path, identifier, discard=discard)
    # Look at the resulting distributions in a cornerplot
    CORNER_KWARGS = dict(smooth=0.90,title_fmt=".2E")
    fig = corner.corner(sampler.get_chain(discard=discard, thin=2, flat=True), labels=labels, **CORNER_KWARGS)
    for idx,ax in enumerate(fig.get_axes()):
        ax.grid(False)
    plt.show()
    plt.close()

    ######################
    ## Visualize result ##
    ######################

    def draw_fcn(param_dict, samples_dict):
        idx, param_dict['Vf_Ks'] = random.choice(list(enumerate(samples_dict['Vf_Ks'])))
        param_dict['R_AS'] = samples_dict['R_AS'][idx]
        param_dict['R_AW'] = samples_dict['R_AW'][idx]
        param_dict['R_Es'] = samples_dict['R_Es'][idx]
        param_dict['K_eq'] = samples_dict['K_eq'][idx]
        return param_dict

    # Loop over datasets
    for i,df in enumerate(data):
        # Update initial condition
        model.initial_states.update(initial_states[i])
        # Simulate model
        out = model.sim(3000, N=N, draw_function=draw_fcn, samples=samples_dict)
        # Add 5% observational noise
        out = add_gaussian_noise(out, 0.05, relative=True)
        # Visualize
        fig,ax=plt.subplots(figsize=(12,4))
        ax.scatter(df.index, df.values, color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
        for i in range(N):
            ax.plot(out['time'], out['S'].isel(draws=i), color='black', alpha=0.03, linewidth=0.2)
            ax.plot(out['time'], out['Es'].isel(draws=i), color='red', alpha=0.03, linewidth=0.2)
        ax.legend(['data', 'D-glucose', 'Glucose laurate'])
        ax.grid(False)
        ax.set_ylabel('species concentration (mM)')
        ax.set_xlabel('time (min)')
        plt.tight_layout()
        plt.show()
        plt.close()
