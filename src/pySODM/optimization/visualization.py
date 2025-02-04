import math
import matplotlib.pyplot as plt
import numpy as np
import emcee

def _apply_tick_locator(ax):
    """support function to apply default ticklocator settings"""
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    return ax

def autocorrelation_plot(samples, labels=None, filename=None, plt_kwargs={}):
    """
    Make a visualization of autocorrelation of each chain

    Parameters
    ----------
    samples: np.array
        A 3-D numpy array containing the sampled parameters.
        The x-dimension must be the number of samples, the y-dimension the number of parallel chains and the z-dimension the number of sampled parameters.

    Returns
    -------
    ax: matplotlib axis object

    tau_vect[-1]: list
        Autocorrelation of last step
    """

    # Extract dimensions of sampler output
    _, _, ndim = samples.shape
    # Generate list of lables if none provided
    if not labels:
        labels = [f"$\\theta_{i}$" for i in range(ndim)]
    else:
        # input check
        if len(labels) != ndim:
            raise ValueError(
            "The length of label list is not equal to the length of the z-dimension of the samples.\n"
            "The list of label is of length: {0}. The z-dimension of the samples of length: {1}".format(len(labels), ndim)
            )

    # Compute autocorrelation/chain
    ndim = samples.shape[2]
    step_autocorr = math.ceil(samples.shape[0]/100)
    tau_vect = []
    index = 0
    for i in range(step_autocorr, samples.shape[0], step_autocorr):
        tau_vect.append(emcee.autocorr.integrated_time(samples[:i], tol = 0))
        index += 1
    n = step_autocorr * np.arange(1, index + 1)

    # Make figure
    _,ax=plt.subplots(figsize=(10,4))
    # Autocorrelation
    ax.plot(n, np.array(tau_vect), **plt_kwargs)
    ax.plot(n, n/50, "--k")
    ax.set_xlim(0, n.max())
    ax.set_xlabel("Iteration (-)")
    ax.set_ylabel("Integrated autocorrelation (-)");    
    ax.grid(False)
    ax.legend(labels)

    # Save result if desired
    if filename:
        plt.savefig(filename, dpi=200, bbox_inches='tight', orientation='portrait')

    return ax, tau_vect[-1]

def traceplot(samples, labels=None, filename=None, plt_kwargs={}):
    """
    Make a visualization of sampled parameters

    Parameters
    ----------
    samples: np.array
        A 3-D numpy array containing the sampled parameters.
        The x-dimension must be the number of samples, the y-dimension the number of parallel chains and the z-dimension the number of sampled parameters.
    labels: list
        A list containing the names of the sampled parameters. Must be the same length as the z-dimension of the samples np.array.
    plt_kwargs: dictionary
        A dictionary containing arguments for the `ax.plot()` function used to visualise the traces.

    Returns
    -------
    ax
    """
    # Extract dimensions of sampler output
    nsamples, _, ndim = samples.shape
    # Generate list of lables if none provided
    if not labels:
        labels = [f"$\\theta_{i}$" for i in range(ndim)]
    else:
        # input check
        if len(labels) != ndim:
            raise ValueError(
            "The length of label list is not equal to the length of the z-dimension of the samples.\n"
            "The list of label is of length: {0}. The z-dimension of the samples of length: {1}".format(len(labels), ndim)
            )

    # initialise figure
    _, axes = plt.subplots(nrows=len(labels), ncols=2, figsize=(8.3,(11.7/6)*len(labels)), width_ratios=[2.5,1])

    # visualise data
    axes =  np.atleast_2d(axes) # np.atleast_2d() solves error if number of parameters equal to one
    for i in range(ndim):
        ax = axes[i,:] 
        # traces
        ax[0].plot(samples[:, :, i], **plt_kwargs)
        ax[0].set_xlim(0, nsamples)
        ax[0].set_ylabel(labels[i])
        ax[0].grid(False)
        # marginal distribution
        d = np.random.choice(samples[:, :, i].flatten(), int(min(len(samples[:, :, i].flatten())/3, 1000)))
        ax[1].hist(d, color='black', density=True)
        ax[1].axvline(np.median(d), color='red', linewidth=1, linestyle='--')

    # axes labels
    axes[-1,0].set_xlabel("Iteration (-)")

    # Save result if desired
    if filename:
        plt.savefig(filename, dpi=200, bbox_inches='tight', orientation='portrait')

    return ax