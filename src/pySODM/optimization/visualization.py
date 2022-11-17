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
    """Make a visualization of autocorrelation of each chain

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
    nsamples,nwalkers, ndim = samples.shape
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
    fig,ax=plt.subplots(figsize=(10,4))
    # Autocorrelation
    ax.plot(n, np.array(tau_vect))
    ax.plot(n, n/50, "--k")
    ax.set_xlim(0, n.max())
    ax.set_ylabel(r"$\hat{\tau}$");    
    ax.grid(False)
    ax.legend(labels)

    # Save result if desired
    if filename:
        plt.savefig(filename, dpi=600, bbox_inches='tight', orientation='portrait')

    return ax, tau_vect[-1]

def traceplot(samples, labels=None, filename=None, plt_kwargs={}):
    """Make a visualization of sampled parameters

    Parameters
    ----------
    samples: np.array
        A 3-D numpy array containing the sampled parameters.
        The x-dimension must be the number of samples, the y-dimension the number of parallel chains and the z-dimension the number of sampled parameters.
    labels: list
        A list containing the names of the sampled parameters. Must be the same length as the z-dimension of the samples np.array.
    plt_kwargs: dictionary
        A dictionary containing arguments for the plt.plot function.

    Returns
    -------
    ax
    """
    # Extract dimensions of sampler output
    nsamples,nwalkers, ndim = samples.shape
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
    fig, axes = plt.subplots(len(labels))
    # Error when only one parameter is calibrated: axes not suscribable
    if ndim == 1:
        axes = [axes]
    # set size
    fig.set_size_inches(10, len(labels)*7/3)
    # plot data
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], **plt_kwargs)
        ax.set_xlim(0, nsamples)
        ax.set_ylabel(labels[i])
        ax.grid(False)
    axes[-1].set_xlabel("step number")

    # Save result if desired
    if filename:
        plt.savefig(filename, dpi=600, bbox_inches='tight', orientation='portrait')

    return ax