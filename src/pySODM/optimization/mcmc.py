import os
import gc
import sys
import emcee
import datetime
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from multiprocessing import get_context
from typing import List, Tuple, Union, Callable, Optional, Dict, Any
from pySODM.optimization.visualization import traceplot, autocorrelation_plot

abs_dir = os.path.dirname(__file__)

def run_EnsembleSampler(
    pos: np.ndarray,  
    max_n: int,  
    identifier: str,  
    objective_function: Callable[[Union[List, np.ndarray], ...], float],  
    objective_function_args: Optional[Tuple] = None,  
    objective_function_kwargs: Optional[Dict[str, Any]] = None,  
    moves: List[Tuple[emcee.moves.Move, float]] = [
        (emcee.moves.DEMove(), 0.5*0.5*0.9), 
        (emcee.moves.DEMove(gamma0=1.0), 0.5*0.5*0.1),
        (emcee.moves.DESnookerMove(), 0.5*0.5),
        (emcee.moves.KDEMove(bw_method='scott'), 0.25),
        (emcee.moves.StretchMove(live_dangerously=True), 0.25)
    ],  
    fig_path: Optional[str] = None,  
    samples_path: Optional[str] = None,  
    print_n: int = 10,  
    discard: int = 0,
    thin: int = 1,
    backend: Optional[str] = None,  
    processes: int = 1,  
    progress: bool = True,  
    settings_dict: Dict[str, Any] = {}  
) -> Tuple[emcee.EnsembleSampler, xr.Dataset]:

    """
    Wrapper function to setup an `emcee.EnsembleSampler` and handle all backend-related tasks.
    
    Parameters:
    -----------
        - pos: np.ndarray
            - Starting position of the Markov Chains. We recommend using `perturbate_theta()`.
        - max_n: int
            - Maximum number of iterations.
        - identifier: str
            - Identifier of the expirement.
        - objective function: callable function
            - Objective function. Recommended `log_posterior_probability`.
        - objective_function_args: tuple
            - Arguments of the objective function. If using `log_posterior_probability` as objective function, use default `None`.
        - objective_function_kwargs: dict
            - Keyworded arguments of the objective function. If using `log_posterior_probability` as objective function, use default `None`.
        - fig_path: str
            - Location where the traceplot and autocorrelation plots should be saved.
        - samples_path: str
            - Location where the `.hdf5` backend and settings `.json` should be saved.
        - print_n: int
            - Print autocorrelation and trace plots every `print_n` iterations.
        - discard: int
            - Number of iterations to remove from the beginning of the markov chains ("burn-in").
        - thin: int
            - Retain only every `thin`-th iteration.
        - processes: int
            - Number of cores to use.
        - settings_dict: dict
            - Dictionary containing calibration settings or other usefull settings for long-term storage. Appended to output `samples_xr` as attributes.
            - Valid datatypes for values: (single values) int, float, str, bool. (arrays) homogeneous list of int, float, str, bool. 1D numpy array containing int, float.

    Hyperparameters:
    ----------------
        - moves: list
            - Algorithm used for updating the coordinates of walkers in an ensemble sampler. By default, pySODM uses a shotgun approach by implementing a balanced cocktail of `emcee` moves. Consult the [emcee documentation](https://emcee.readthedocs.io/en/stable/user/moves/).
        - backend: str
            - Location of backend previous sampling (samples_path + backend). If a backend is provided, the sampler is restarted from the last iteration of the previous run. Consult the [emcee documentation](https://emcee.readthedocs.io/en/stable/user/backends/).
        - progress: bool
            - Enables the progress bar.

    Returns:
    --------

    - sampler: emcee.EnsembleSampler
        - Emcee sampler object ([see](https://emcee.readthedocs.io/en/stable/user/sampler/)).
    
    - samples_xr: xarray.Dataset
        - samples formatted in an xarray.Dataset
        - scalar parameters:
            - dimensions: ['iteration', 'chain']
            - coordinates: [samples_np.shape[0], samples_np.shape[1]]
        - n-D  parameters:
            - dimensions: ['iteration', 'chain', '{parname}_dim_0', ..., '{parname}_dim_n']
            - coordinates: [samples_np.shape[0], samples_np.shape[1], parameter_shapes[parname][0], ..., parameter_shapes[parname][n]]
    """

    # Set default fig_path/samples_path as same directory as calibration script
    if not fig_path:
        fig_path = os.getcwd() + '/'
    else:
        if fig_path[-1] != '/':
            fig_path = fig_path+'/'
        fig_path = os.path.join(os.getcwd(), fig_path)
        # If it doesn't exist make it
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
    if not samples_path:
        samples_path = os.getcwd() + '/'
    else:
        if samples_path[-1] != '/':
            samples_path = samples_path+'/'
        samples_path = os.path.join(os.getcwd(), samples_path)
        # If it doesn't exist make it
        if not os.path.exists(samples_path):
            os.makedirs(samples_path)
    # Determine current date
    run_date = str(datetime.date.today())
    # Derive nwalkers, ndim from shape of pos
    nwalkers, ndim = pos.shape
    # Append thin and discard to settings dictionary
    settings_dict.update({'discard': discard, 'thin': thin})
    # Validate the types in the settings dictionary --> otherwise error dumping to xarray
    _validate_settings_dict(settings_dict)
    # Start printout for user
    print(f'\nMarkov-Chain Monte-Carlo sampling')
    print(f'=================================\n')   
    # By default: set up a fresh hdf5 backend in samples_path
    if not backend:
        fn_backend = str(identifier)+'_BACKEND_'+run_date+'.hdf5'
        backend = emcee.backends.HDFBackend(samples_path+fn_backend)
        backend.reset(nwalkers, ndim)

        print(f"Created new backend: '{samples_path+fn_backend}'\n")
        print(f"Starting new run")
        print(f"----------------\n")
        print(f"Parameters: {ndim}")
        print(f"Markov chains: {nwalkers}")
        print(f"Cores: {processes}")
        print(f"Iterations: {max_n}")
        print(f"Automatically checking convergence every {print_n} iterations")
        print(f"Saving samples in an xarray.Dataset every {print_n} iterations")
        print(f"Printing traceplot and autocorrelation plot every {print_n} iterations")
        print(f"Samples: {samples_path+identifier+'_SAMPLES_'+run_date+'.nc'}")
        print(f"Traceplot: {fig_path+identifier+'_TRACE_'+run_date+'.pdf'}")
        print(f"Autocorrelation plot: {fig_path+identifier+'_AUTOCORR_'+run_date+'.pdf'}\n")

    # If user provides an existing backend: continue sampling 
    else:
        backend_path=os.path.join(os.getcwd(), samples_path+backend)
        try:
            backend = emcee.backends.HDFBackend(backend)
            pos = backend.get_chain(discard=0, thin=1, flat=False)[-1, ...]
        except:
            raise FileNotFoundError("backend not found.")    
        
        print(f"Found existing backend:'{backend_path}'\n")
        print(f"Continuing run")
        print(f"--------------\n")
        print(f"Parameters: {ndim}")
        print(f"Markov chains: {nwalkers}")
        print(f"Cores: {processes}")
        print(f"Iterations: {max_n} (found {backend.get_chain(discard=0, thin=1, flat=False).shape[0]} previous iterations)")
        print(f"Automatically checking convergence every {print_n} iterations")
        print(f"Saving samples in an xarray.Dataset every {print_n} iterations")
        print(f"Printing traceplot and autocorrelation plot every {print_n} iterations")
        print(f"Samples: {samples_path+identifier+'_SAMPLES_'+run_date+'.nc'}")
        print(f"Traceplot: {fig_path+identifier+'_TRACE_'+run_date+'.pdf'}")
        print(f"Autocorrelation plot: {fig_path+identifier+'_AUTOCORR_'+run_date+'.pdf'}\n")
    sys.stdout.flush()

    # This will be useful to testing convergence
    old_tau = np.inf

    with get_context("spawn").Pool(processes=processes) as pool:

        # setup sampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, objective_function, backend=backend, pool=pool,
                        args=objective_function_args, kwargs=objective_function_kwargs, moves=moves)
        
        # deduce starting iteration
        sampler_iteration_0 = sampler.iteration 

        # run sampler
        for _ in sampler.sample(pos, iterations=max_n, progress=progress, store=True, tune=False):

            # Only automatically check convergence + printouts every `print_n` steps
            if not ((sampler.iteration > 0 and sampler.iteration % print_n == 0) or (sampler_iteration_0 + sampler.iteration == max_n)):
                continue

            ###############################
            ## UPDATE DIAGNOSTIC FIGURES ##
            ###############################
            
            # Update autocorrelation plot
            _, tau = autocorrelation_plot(sampler.get_chain(), labels=objective_function.expanded_labels,
                                            filename=fig_path+identifier+'_AUTOCORR_'+run_date+'.pdf',
                                            plt_kwargs={})
            # Update traceplot
            traceplot(sampler.get_chain(),labels=objective_function.expanded_labels,
                        filename=fig_path+identifier+'_TRACE_'+run_date+'.pdf',
                        plt_kwargs={'linewidth': 1,'color': 'black','alpha': 0.2})
            # Garbage collection
            plt.close('all')
            gc.collect()

            ###################################
            ## SAVE SAMPLES IN XARRAY FORMAT ##
            ###################################

            samples_xr = _dump_sampler_to_xarray(sampler.get_chain(discard=discard, thin=thin), objective_function.parameter_shapes,
                                                    filename=samples_path+identifier+'_SAMPLES_'+run_date+'.nc', settings_dict=settings_dict)

            #######################
            ## CHECK CONVERGENCE ##
            #######################

            # Hardcode threshold values defining convergence
            thres_multi = 50.0
            thres_frac = 0.03
            # Check if chain length > 50*max autocorrelation
            converged = np.all(np.max(tau) * thres_multi < sampler.iteration)
            # Check if average tau varied more than three percent
            converged &= np.all(np.abs(np.mean(old_tau) - np.mean(tau)) / np.mean(tau) < thres_frac)
            # Update tau
            old_tau = tau
            if converged:
                print(f'Convergence: The chain is longer than 50 times the intergrated autocorrelation time.\n')
                sys.stdout.flush()
                break 
            else:
                print(f'Non-convergence: The chain is shorter than 50 times the integrated autocorrelation time.\n')
                sys.stdout.flush()

    return sampler, samples_xr


def perturbate_theta(theta: Union[List[float], np.ndarray],
                     pert: Union[List[float], np.ndarray],
                     multiplier: int=2,
                     bounds: Optional[List[Tuple[float, float]]] = None,
                     verbose: Optional[bool]=None) -> Tuple[int, int, np.ndarray]:
    """ A function to perturbate a PSO estimate and construct a matrix with initial positions for the MCMC chains

    Parameters
    ----------

    - theta : list (containing floats) or np.array
        - Parameter estimate.

    - pert : list (containing floats) or np.array
        - Relative perturbation factors (plus-minus) on parameter estimate. Must have the same length as `theta`.

    - multiplier : int
        - Multiplier determining the total number of markov chains that will be run by emcee.
        - Typically, total nr. chains = multiplier * nr. parameters
        - Default (minimum): 2 (one chain will result in an error in emcee)
        
    - bounds : array of tuples of floats
        - Ordered boundaries for the parameter values, e.g. ((0.1, 1.0), (1.0, 10.0)) if there are two parameters.
        - Note: bounds must not be zero, because the perturbation is based on a percentage of the value, and any percentage of zero returns zero, causing an error regarding linear dependence of walkers
        
    - verbose : boolean
        - Print user feedback to stdout

    Returns
    -------
    - ndim : int
        - Number of parameters

    - nwalkers : int
        - Number of chains

    - pos : np.array
        - Initial positions for markov chains. Dimensions: [nwalkers, ndim]
    """

    # Validation
    if len(theta) != len(pert):
        raise Exception('The parameter value array "theta" must have the same length as the perturbation value array "pert".')
    if len(bounds) != len(theta):
        raise Exception('If bounds is not None, it must contain a tuple for every parameter in theta')
    # Convert theta to np.array
    theta = np.array(theta)
    # Define clipping values: perturbed value must not fall outside this range
    lower_bounds = [bounds[i][0]/(1-pert[i]) for i in range(len(bounds))]
    upper_bounds = [bounds[i][1]/(1+pert[i]) for i in range(len(bounds))]
    # Start loop
    ndim = len(theta)
    nwalkers = ndim*multiplier
    cond_number=np.inf
    retry_counter=0
    while cond_number == np.inf:
        if retry_counter==0:
            theta = np.clip(theta, lower_bounds, upper_bounds)
        pos = theta + theta*pert*np.random.uniform(low=-1,high=1,size=(nwalkers,ndim))
        cond_number = np.linalg.cond(pos)
        if ((cond_number == np.inf) and verbose and (retry_counter<20)):
            print("Condition number too high, recalculating perturbations. Perhaps one or more of the bounds is zero?")
            sys.stdout.flush()
            retry_counter += 1
        elif retry_counter >= 20:
            raise Exception("Attempted 20 times to perturb parameter values but the condition number remains too large.")
    if verbose:
        print('Total number of markov chains: ' + str(nwalkers)+'\n')
        sys.stdout.flush()
    return ndim, nwalkers, pos


def _dump_sampler_to_xarray(samples_np: np.ndarray, parameter_shapes: Dict[str, Tuple], filename: str=None, settings_dict: Dict=None) -> xr.Dataset:
    """
    A function converting the raw samples from `emcee` (numpy matrix) to an xarray dataset for convenience

    Parameters
    ----------

    - samples_np: np.ndarray
        - 3D numpy array, indexed: iteration, markov chain, parameter. 
        - obtained using `sampler.get_chain()`

    - parameter_shapes: dict
        - keys: parameter name, value: parameter shape (type: tuple)

    - filename: str (optional)
        - path and filename to store samples in (default: None -- samples not saved)

    - settings_dict: dict
        - contains calibration settings retained for long-term storage. appended as metadata to the xarray output.
        - valid datatypes for values: str, Number, ndarray, number, list, tuple, bytes
    
    Returns
    -------

    - samples_xr: xarray.Dataset
        - samples formatted in an xarray.Dataset
        - scalar parameters:
            - dimensions: ['iteration', 'chain']
            - coordinates: [samples_np.shape[0], samples_np.shape[1]]
        - n-D  parameters:
            - dimensions: ['iteration', 'chain', '{parname}_dim_0', ..., '{parname}_dim_n']
            - coordinates: [samples_np.shape[0], samples_np.shape[1], parameter_shapes[parname][0], ..., parameter_shapes[parname][n]]
    """

    data = {}
    count=0
    for name,shape in parameter_shapes.items():
        # pre-allocate dims and coords
        dims = ['iteration', 'chain']
        coords = {'iteration': range(samples_np.shape[0]), 'chain': range(samples_np.shape[1])}
        # multi-dimensional parameter
        if shape != (1,):
            # samples
            param_samples = samples_np[:, :, count : count + np.prod(shape)]                            # extract relevant slice
            reshaped_samples = param_samples.reshape(samples_np.shape[0], samples_np.shape[1], *shape)  # reshape into nD array
            arr = reshaped_samples
            count += np.prod(shape)
            # dimensions and coordinates
            for i, len_dim in enumerate(shape):
                dims.append(f'{name}_dim_{i}')
                coords[f'{name}_dim_{i}'] = range(len_dim)
        # scalar parameter
        else:  # Scalar parameter case
            arr = samples_np[:, :, count]  # Keep as 2D array
            count += 1
        # wrap in an xarray
        data[name] = xr.DataArray(arr, dims=dims, coords=coords)

    # combine it all
    samples_xr = xr.Dataset(data)

    # append settings as attributes
    samples_xr.attrs.update(settings_dict)

    # save it
    if filename:
        samples_xr.to_netcdf(filename)

    return samples_xr

def _validate_settings_dict(settings_dict: Dict) -> None:
    """
    Checks if the entries in `settings_dict` are compatible for export to netCDF.
    
    Valid types are:
        - Scalar values: int, float, bool, str
        - Arrays: homogeneous lists of int, float, bool, str; 1D NumPy arrays containing int, float
    """

    def is_homogeneous_list(lst):
        """Check if list is homogeneous and contains only allowed types."""
        if not isinstance(lst, list) or not lst:
            return False
        first_type = type(lst[0])
        return all(isinstance(item, first_type) for item in lst) and first_type in {int, float, str, bool}

    def is_valid_numpy_array(arr):
        """Check if NumPy array is 1D and contains only int or float values."""
        return isinstance(arr, np.ndarray) and arr.ndim == 1 and arr.dtype.kind in {'i', 'f'}

    for key, value in settings_dict.items():
        if isinstance(value, (int, float, bool, str, list, np.ndarray)):
            if isinstance(value, list):
                if not is_homogeneous_list(value):
                    raise TypeError(f"Invalid datatype in `settings_dict` for key: '{key}' (type: {type(value)}). lists must be homogeneous and contain only int, float, bool or str.")
            elif isinstance(value, np.ndarray):
                if not is_valid_numpy_array(value):
                    raise TypeError(f"Invalid datatype in `settings_dict` for key: '{key}' (type: {type(value)}). numpy arrays must be 1D and contain only int or float.")
        else:
            raise TypeError(f"Invalid datatype in `settings_dict` for key: '{key}' (type: {type(value)}).\nvalid types are: (scalar values) int, float, bool, str. (arrays) homogeneous lists containing int, float, str, bool. 1D numpy arrays containing int or float.")
    pass