import os
import gc
import sys
import emcee
import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import get_context
from pySODM.optimization.visualization import traceplot, autocorrelation_plot

abs_dir = os.path.dirname(__file__)

def run_EnsembleSampler(pos, max_n, identifier, objective_fcn, objective_fcn_args, objective_fcn_kwargs,
                moves=[(emcee.moves.DEMove(), 0.5),(emcee.moves.KDEMove(bw_method='scott'), 0.5)],
                fig_path=None, samples_path=None, print_n=10, labels=None, backend=None, processes=1, progress=True, settings_dict={}):

    # Set default fig_path/samples_path as same directory as calibration script
    if not fig_path:
        fig_path = os.getcwd()
    else:
        fig_path = os.path.join(os.getcwd(), fig_path)
    if not samples_path:
        samples_path = os.getcwd()
    else:
        samples_path = os.path.join(os.getcwd(), samples_path)
    # Check if the fig_path/autocorrelation and fig_path/traceplots exist and if not make them
    for directory in [fig_path+"/autocorrelation/", fig_path+"/traceplots/"]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    # Determine current date
    run_date = str(datetime.date.today())
    # Save setings dictionary to samples_path
    with open(samples_path+'/'+str(identifier)+'_SETTINGS_'+run_date+'.json', 'w') as file:
        json.dump(settings_dict, file)
    # Derive nwalkers, ndim from shape of pos
    nwalkers, ndim = pos.shape
    # By default: set up a fresh hdf5 backend in samples_path
    if not backend:
        filename = '/'+str(identifier)+'_BACKEND_'+run_date+'.h5'
        backend = emcee.backends.HDFBackend(samples_path+filename)
        backend.reset(nwalkers, ndim)
    # If user provides an existing backend: continue sampling 
    else:
        pos = backend.get_chain(discard=0, thin=1, flat=False)[-1, ...]
    # This will be useful to testing convergence
    old_tau = np.inf

    with get_context("spawn").Pool(processes=processes) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, objective_fcn, backend=backend, pool=pool,
                        args=objective_fcn_args, kwargs=objective_fcn_kwargs, moves=moves)
        for sample in sampler.sample(pos, iterations=max_n, progress=progress, store=True, tune=True):
            # Only check convergence every print_n steps
            if sampler.iteration % print_n:
                continue

            #############################
            # UPDATE DIAGNOSTIC FIGURES #
            #############################
            
            # Update autocorrelation plot
            ax, tau = autocorrelation_plot(sampler.get_chain(), labels=labels,
                                            filename=fig_path+'/autocorrelation/'+identifier+'_AUTOCORR_'+run_date+'.pdf',
                                            plt_kwargs={'linewidth':2, 'color': 'red'})
            # Update traceplot
            traceplot(sampler.get_chain(),labels=labels,
                        filename=fig_path+'/traceplots/'+identifier+'_TRACE_'+run_date+'.pdf',
                        plt_kwargs={'linewidth':2,'color': 'red','alpha': 0.15})
            # Garbage collection
            plt.close('all')
            gc.collect()

            #####################
            # CHECK CONVERGENCE #
            #####################

            # Hardcode threshold values defining convergence
            thres_multi = 50.0
            thres_frac = 0.03
            # Check convergence using mean tau
            converged = np.all(np.mean(tau) * thres_multi < sampler.iteration)
            converged &= np.all(np.abs(np.mean(old_tau) - np.mean(tau)) / np.mean(tau) < thres_frac)
            if converged:
                break
            old_tau = tau

            #################################
            # LEGACY: WRITE SAMPLES TO .NPY #
            #################################

            # Write samples to dictionary every print_n steps
            #if sampler.iteration % print_n:
            #    continue

            #if not progress:
            #    print(f"Saving samples as .npy file for iteration {sampler.iteration}/{max_n}.")
            #    sys.stdout.flush()
                
            #flat_samples = sampler.get_chain(flat=True)
            #with open(samples_path+'/'+str(identifier)+'_SAMPLES_'+run_date+'.npy', 'wb') as f:
            #    np.save(f,flat_samples)
            #    f.close()
            #    gc.collect()

    return sampler

def emcee_sampler_to_dictionary(sampler, parameter_names, discard=0, thin=1, settings={}):
    """
    A function to discard and thin the samples available in the sampler object. Convert them to a dictionary of format: {parameter_name: [sample_0, ..., sample_n]}.
    Append a dictionary of settings (f.i. starting estimate of MCMC sampler, start- and enddate of calibration).
    """
    ####################
    # Discard and thin #
    ####################

    thin = 1
    try:
        autocorr = sampler.get_autocorr_time()
        thin = max(1,int(0.5 * np.min(autocorr)))
        print(f'Convergence: the chain is longer than 50 times the intergrated autocorrelation time.\nPreparing to save samples with thinning value {thin}.')
        sys.stdout.flush()
    except:
        print('Warning: The chain is shorter than 50 times the integrated autocorrelation time.\nUse this estimate with caution and run a longer chain! Setting thinning to 1.\n')
        sys.stdout.flush()

    #####################################
    # Construct a dictionary of samples #
    #####################################

    # Samples
    flat_samples = sampler.get_chain(discard=discard,thin=thin,flat=True)
    samples_dict = {}
    for count,name in enumerate(parameter_names):
        samples_dict[name] = flat_samples[:,count].tolist()
    
    # Append settings
    samples_dict.update(settings)

    return samples_dict