import sys
import numpy as np
from functools import partial

def _obj_wrapper(func, args, kwargs, x):
    return -func(x, *args, **kwargs)


def _is_feasible_wrapper(func, x):
    return np.all(func(x) >= 0)


def _cons_none_wrapper(x):
    return np.array([0])


def _cons_ieqcons_wrapper(ieqcons, args, kwargs, x):
    return np.array([y(x, *args, **kwargs) for y in ieqcons])


def _cons_f_ieqcons_wrapper(f_ieqcons, args, kwargs, x):
    return np.array(f_ieqcons(x, *args, **kwargs))


def optimize(func, bounds=None, ieqcons=[], f_ieqcons=None, args=(), kwargs={},
        processes=1, swarmsize=100, max_iter=100, minstep=1e-12, minfunc=1e-12, omega=0.8, phip=0.8, phig=0.8, 
        debug=False, particle_output=False, transform_pars=None):
    """
    Perform a particle swarm optimization (PSO)

    Parameters
    ==========
    func : callable function or class 'log_posterior_probability' (~/src/optimization/objective_functions.py)
        The objective function to be minimized
    bounds: tuple array
        The bounds of the design variable(s). In form [(lower, upper), ..., (lower, upper)]
        Class 'log_posterior_probability' automatically contains bounds. If bounds are provided these overwrite the bounds available in the 'log_posterior_probability' object.

    Optional
    ========
    ieqcons : list
        A list of functions of length n such that ieqcons[j](x,*args) >= 0.0 in 
        a successfully optimized problem (Default: [])
    f_ieqcons : function
        Returns a 1-D array in which each element must be greater or equal 
        to 0.0 in a successfully optimized problem. If f_ieqcons is specified, 
        ieqcons is ignored (Default: None)
    args : tuple
        Additional arguments passed to objective function
        (Default: empty tuple)
    kwargs : dict
        Additional keyword arguments passed to objective function
    phig : scalar
        Scaling factor to search away from the swarm's best known position
        (Default: 0.5)
    max_iter : int
        The maximum number of iterations for the swarm to search (Default: 100)
    minstep : scalar
        The minimum stepsize of swarm's best position before the search
        terminates (Default: 1e-8)
    minfunc : scalar
        The minimum change of swarm's best objective value before the search
        terminates (Default: 1e-8)
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
    processes : int
        The number of processes to use to evaluate objective function and 
        constraints (default: 1)
    particle_output : boolean
        Whether to include the best per-particle position and the objective
        values at those.
    transform_pars : None / function
        Transform the parameter values. E.g. to integer values or to map to
        a list of possibilities.

    Returns
    =======
    g : array
        The swarm's best known position (optimal design)
    f : scalar
        The objective value at ``g``
    p : array
        The best known position per particle
    pf: arrray
        The objective values at each position in p

    """

    if not bounds:
        try:
            bounds = func.expanded_bounds
        except:
            raise Exception(
                "'func' does not appear to be a pySODM model: 'expanded_bounds' not found. Provide bounds directly to `pso.optimize()`"
            )

    lb, ub = [], []
    for variable_bounds in bounds:
        lb.append(variable_bounds[0])
        ub.append(variable_bounds[1])

    assert len(lb) == len(ub), 'Lower- and upper-bounds must be the same length'
    assert hasattr(func, '__call__'), 'Invalid function handle'
    lb = np.array(lb)
    ub = np.array(ub)
    assert np.all(ub > lb), 'All upper-bound values must be greater than lower-bound values'

    vhigh = np.abs(ub - lb)
    vlow = -vhigh

    # Initialize objective function.
    # The only remaining argument for obj(thetas) is thetas, a vector containing estimated parameter values
    # these values thetas will be based on the PSO dynamics and the boundary conditions in lb and ub.
    obj = partial(_obj_wrapper, func, args, kwargs)

    print(f'\nParticle Swarm minimization')
    print(f'===========================\n')

    print(f'Using {processes} cores')
    
    # Check for constraint function(s) #########################################
    if f_ieqcons is None:
        if not len(ieqcons):
            if debug:
                print('Without constraints')
                sys.stdout.flush()
            cons = _cons_none_wrapper
        else:
            if debug:
                print('Converting ieqcons to a single constraint function')
                sys.stdout.flush()
            cons = partial(_cons_ieqcons_wrapper, ieqcons, args, kwargs)
    else:
        if debug:
            print('Single constraint function given in f_ieqcons')
            sys.stdout.flush()
        cons = partial(_cons_f_ieqcons_wrapper, f_ieqcons, args, kwargs)

    is_feasible = partial(_is_feasible_wrapper, cons)

    print(f'Using the following bounds: {bounds}\n')

    # Initialize the multiprocessing module if necessary
    if processes > 1:
        import multiprocessing
        mp_pool = multiprocessing.Pool(processes)
        

    # Initialize the particle swarm ############################################
    S = swarmsize
    D = len(lb)  # the number of dimensions each particle has
    x = np.random.rand(S, D)  # particle positions
    v = np.zeros_like(x)  # particle velocities
    p = np.zeros_like(x)  # best particle positions
    fx = np.zeros(S)  # current particle function values
    fs = np.zeros(S, dtype=bool)  # feasibility of each particle
    fp = np.ones(S)*np.inf  # best particle function values
    g = []  # best swarm position

    fg = np.inf  # best swarm position starting value

    # Initialize the particle's position
    x = lb + x*(ub - lb)

    # if needed, transform the parameter vector
    if transform_pars is not None:
        x = np.apply_along_axis(transform_pars, 1, x)
        
    # Calculate objective and constraints for each particle
    if processes > 1:
        fx = np.array(mp_pool.map(obj, x))
        fs = np.array(mp_pool.map(is_feasible, x))
    else:
        for i in range(S):
            fx[i] = obj(x[i, :])
            fs[i] = is_feasible(x[i, :])

    # Store particle's best position (if constraints are satisfied)
    i_update = np.logical_and((fx < fp), fs)
    p[i_update, :] = x[i_update, :].copy()
    fp[i_update] = fx[i_update]

    # Update swarm's best position
    i_min = np.argmin(fp)
    if fp[i_min] < fg:
        fg = fp[i_min]
        g = p[i_min, :].copy()
    else:
        # At the start, there may not be any feasible starting point, so just
        # give it a temporary "best" point since it's likely to change
        g = x[0, :].copy()
       
    # Initialize the particle's velocity
    v = vlow + np.random.rand(S, D)*(vhigh - vlow)

    # Iterate until termination criterion met ##################################
    it = 1

    while it <= max_iter:
        rp = np.random.uniform(size=(S, D))
        rg = np.random.uniform(size=(S, D))
        # Update the particles velocities
        v = omega*v + phip*rp*(p - x) + phig*rg*(g - x)
        # Update the particles' positions
        x = x + v
        # Correct for bound violations
        maskl = x < lb
        masku = x > ub
        x = x*(~np.logical_or(maskl, masku)) + lb*maskl + ub*masku
        # if needed, transform the parameter vector
        if transform_pars is not None:
            x = np.apply_along_axis(transform_pars, 1, x)


        # Update objectives and constraints
        if processes > 1:
            fx = np.array(mp_pool.map(obj, x))
            fs = np.array(mp_pool.map(is_feasible, x))
        else:
            for i in range(S):
                fx[i] = obj(x[i, :])
                fs[i] = is_feasible(x[i, :])

        # Store particle's best position (if constraints are satisfied)
        i_update = np.logical_and((fx < fp), fs)
        p[i_update, :] = x[i_update, :].copy()
        fp[i_update] = fx[i_update]

        # Compare swarm's best position with global best position
        i_min = np.argmin(fp)
        if fp[i_min] < fg:
            if debug:
                print('New best at iteration {:}: {:.3e}; {:} '\
                    .format(it, fp[i_min], p[i_min, :],))
                sys.stdout.flush()
            p_min = p[i_min, :].copy()
            stepsize = np.sqrt(np.sum((g - p_min)**2))

            if np.abs(fg - fp[i_min]) <= minfunc:
                print('Stopping search: Swarm best objective change less than {:}'\
                    .format(minfunc))
                sys.stdout.flush()
                if particle_output:
                    return p_min, fp[i_min], p, fp
                else:
                    return p_min, fp[i_min]
            elif stepsize <= minstep:
                print('Stopping search: Swarm best position change less than {:}'\
                    .format(minstep))
                sys.stdout.flush()
                if particle_output:
                    return p_min, fp[i_min], p, fp
                else:
                    return p_min, fp[i_min]
            else:
                g = p_min.copy()
                fg = fp[i_min]

        if debug:
            print('Best after iteration {:}: {:.3e}; {:}'.format(it, fg, g))
            sys.stdout.flush()
        it += 1

    print('Stopping search: maximum iterations reached --> {:}'.format(max_iter))
    sys.stdout.flush()
    
    if processes > 1:
        mp_pool.close()

    if not is_feasible(g):
        print("However, the optimization couldn't find a feasible design. Sorry")
        sys.stdout.flush()
    if particle_output:
        return g, fg, p, fp
    else:
        return g, fg
