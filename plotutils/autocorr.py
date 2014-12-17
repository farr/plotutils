"""Module containing functions to compute the autocorrelation function
and estimate the associated autocorrelation length of series.

The estimate of the autocorrelation function is based on the method
described at http://www.math.nyu.edu/faculty/goodman/software/acor/
and implemented in the associated ``acor`` C++ code, though the
implementation details differ.

"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

def _next_power_of_two(i):
    pt = 2
    while pt < i:
        pt *= 2

    return pt

def autocorrelation_function(series, axis=0):
    """Returns the autocorrelation function of the given series.  The
    function is normalised so that it is 1 at zero lag.

    If ``series`` is an N-dimensional array, the ACF will be computed
    along ``axis`` and the result will have the same shape as
    ``series``.

    """
    series = np.atleast_1d(series)
    shape = np.array(series.shape)
    m = [slice(None)] * len(shape)

    n0 = shape[axis]
    n = _next_power_of_two(shape[axis]*2)
    m[axis] = slice(0, n0)
    shape[axis] = n

    padded_series = np.zeros(shape)
    padded_series[m] = series - np.expand_dims(series.mean(axis=axis), axis=axis)

    ps_tilde = np.fft.fft(padded_series, axis=axis)
    acf = np.real(np.fft.ifft(ps_tilde*np.conj(ps_tilde), axis=axis))[m]

    m[axis] = 0
    shape[axis] = 1
    acf /= acf[m].reshape(shape).repeat(n0, axis)

    return acf

def autocorrelation_length_estimate(series, acf=None, M=5, axis=0):
    r"""Returns an estimate of the autocorrelation length of the given
    series:

    .. math::

      L = \int_{-\infty}^\infty \rho(t) dt

    The estimate is the smallest :math:`L` such that 

    .. math::

      L = \rho(0) + 2 \sum_{j = 1}^{M L} \rho(j)

    In words: the ACL is estimated over a window that is at least
    :math:`M` ACLs long, with the constraint that :math:`ML < N/2`.

    Defined in this way, the ACL gives the reduction factor between
    the number of samples and the "effective" number of samples.  In
    particular, the variance of the estimated mean of the series is
    given by

    .. math::

      \left\langle \left( \frac{1}{N} \sum_{i=0}^{N-1} x_i - \mu
      \right)^2 \right\rangle = \frac{\left\langle \left(x_i -
      \mu\right)^2 \right\rangle}{N/L}

    Returns ``nan`` if there is no such estimate possible (because
    the series is too short to fit :math:`2M` ACLs).

    For an N-dimensional array, returns an array of ACLs of the same
    shape as ``series``, but with the dimension along ``axis``
    removed.

    """
    if acf is None:
        acf = autocorrelation_function(series, axis=axis)
    m = [slice(None)] * len(acf.shape)
    nmax = acf.shape[axis]/2

    # Generate ACL candidates.
    m[axis] = slice(0, nmax)
    acl_ests = 2.0*np.cumsum(acf[m], axis=axis) - 1.0

    # Build array of lags (like arange, but N-dimensional).
    shape = acf.shape[:axis] + (nmax,) + acf.shape[axis+1:]
    lags = np.cumsum(np.ones(shape), axis=axis) - 1.0

    # Mask out unwanted lags and set corresponding ACLs to nan.
    mask = M*acl_ests >= lags
    acl_ests[mask] = np.nan
    i = ma.masked_greater(mask, lags, copy=False)

    # Now get index of smallest unmasked lag -- if all are masked, this will be 0.
    j = i.argmin(axis=axis)
    k = tuple(np.indices(j.shape))
    return acl_ests[k[:axis] + (j,) + k[axis:]]

def _default_burnin(M):
    return 1.0/(M + 1.0)

def emcee_chain_autocorrelation_lengths(chain, M=5, fburnin=None):
    r"""Returns an array giving the ACL for each parameter in the given
    emcee chain.

    :param chain: The emcee sampler chain.

    :param M: See :func:`autocorrelation_length_estimate`

    :param fburnin: Discard the first ``fburnin`` fraction of the
      samples as burn-in before computing the ACLs.  Default is to
      discard the first :math:`1/(M+1)`, ensuring that at least one
      ACL is discarded.

    """

    if fburnin is None:
        fburnin = _default_burnin(M)

    istart = int(round(fburnin*chain.shape[1]))

    return np.array([autocorrelation_length_estimate(np.mean(chain[:,istart:,k], axis=0)) for k in range(chain.shape[2])])

def emcee_ptchain_autocorrelation_lengths(ptchain, M=5, fburnin=None):
    r"""Returns an array of shape ``(Ntemp, Nparams)`` giving the estimated
    autocorrelation lengths for each parameter across each temperature
    of the parallel-tempered set of chains.  If a particular ACL
    cannot be estimated, that element of the array will be ``None``.
    See :func:`emcee_chain_autocorrelation_lengths` for a description
    of the optional arguments.

    """

    return np.array([emcee_chain_autocorrelation_lengths(ptchain[i,...], M=M, fburnin=fburnin) for i in range(ptchain.shape[0])])

def emcee_thinned_chain(chain, M=5, fburnin=None):
    r"""Returns a thinned, burned-in version of the emcee chain.

    :param chain: The emcee sampler chain.

    :param M: See :func:`autocorrelation_length_estimate`

    :param fburnin: Discard the first ``fburnin`` fraction of the
      samples as burn-in before computing the ACLs.  Default is to
      discard the first :math:`1/(M+1)`, ensuring that at least one
      ACL is discarded.

    """

    if fburnin is None:
        fburnin = _default_burnin(M)

    istart = int(round(fburnin*chain.shape[1]))

    acls = emcee_chain_autocorrelation_lengths(chain, M=M, fburnin=fburnin)

    if any(ac is None for ac in acls):
        return None

    tau = int(np.ceil(np.max(acls)))

    return chain[:,istart::tau,:]

def emcee_thinned_ptchain(ptchain, M=5, fburnin=None):
    r"""Returns a thinned, burned in version of the emcee parallel-tempered
    chains in ``ptchain``, or ``None`` if it is not possible to
    estimate an ACL for some component of the chain.

    """

    if fburnin is None:
        fburnin = _default_burnin(M)

    istart = int(round(fburnin*ptchain.shape[2]))

    acls = emcee_ptchain_autocorrelation_lengths(ptchain, M=M, fburnin=fburnin)

    if any(ac is None for ac in acls.flatten()):
        return None

    tau = int(np.ceil(np.max(acls)))

    return ptchain[:,:,istart::tau,:]

def plot_emcee_chain_autocorrelation_functions(chain, fburnin=None):
    r"""Plots a grid of the autocorrelation function (post burnin) for each
    of the parameters in the given chain.
    
    """

    if fburnin is None:
        fburnin = _default_burnin(5)

    istart = int(round(fburnin*chain.shape[1]))

    chain = chain[:,istart:,:]

    npar = chain.shape[2]
    nside = int(np.ceil(np.sqrt(npar)))

    for i in range(npar):
        plt.subplot(nside, nside, i+1)
        plt.plot(autocorrelation_function(np.mean(chain[:,:,i], axis=0)))
