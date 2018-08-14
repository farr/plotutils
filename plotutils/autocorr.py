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

def crosscorrelation_function(series1, series2, axis=0, returnlags=True):
    """Returns the cross-correlation of the given pair of series.

    If ``series1`` and ``series2`` are of different shapes, then they are first
    broadcast according to the numpy broadcasting rules; ``axis`` refers to the
    corresponding axis of the resulting, broadcasted arrays.

    The returned array will correspond to the correlation comptued holding
    ``series1`` fixed and shifting (i.e. lagging) ``series2`` by ``[-(n-1),
    -(n-2), ..., 0, 1, ..., n-1]`` timesteps, with sufficient zero-padding before
    and after the series, where ``n`` is the length along ``axis``.  If the
    argument ``returnlags`` is ``True``, then ``(ccf, lags)`` will be returned.

    """

    series1 = np.atleast_1d(series1)
    series2 = np.atleast_1d(series2)

    series1, series2 = np.broadcast_arrays(series1, series2)

    shape = np.array(series1.shape)

    m = [slice(None)] * len(shape)
    m2 = [slice(None)] * len(shape)

    n0 = shape[axis]
    n2 = 2*shape[axis]
    n = _next_power_of_two(2*shape[axis])

    m[axis] = slice(0, n0)
    m2[axis] = slice(n-n0+1, n)

    shape[axis] = n

    padded_series1 = np.zeros(shape)
    padded_series1[m] = (series1 - np.expand_dims(series1.mean(axis=axis), axis=axis))/np.std(series1, axis=axis)

    padded_series2 = np.zeros(shape)
    padded_series2[m] = (series2 - np.expand_dims(series2.mean(axis=axis), axis=axis))/np.std(series2, axis=axis)

    ps1_tilde = np.fft.fft(padded_series1, axis=axis)
    ps2_tilde = np.fft.fft(padded_series2, axis=axis)

    ccf = np.real(np.fft.ifft(ps1_tilde*np.conj(ps2_tilde), axis=axis))/n0
    ccf = np.concatenate((ccf[m2], ccf[m]), axis=axis)

    if returnlags:
        return ccf, np.arange(-(n0-1), n0, dtype=np.int)
    else:
        return ccf

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
    m = [slice(None), ] * len(acf.shape)
    nmax = acf.shape[axis]//2

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

    return autocorrelation_length_estimate(np.mean(chain[:,istart:,:], axis=0), axis=0)

def kombine_chain_autocorrelation_lengths(chain, M=5, fburnin=None):
    """Just like :func:`emcee_chain_autocorrelation_lengths` but for kombine.

    """

    return emcee_chain_autocorrelation_lengths(np.transpose(chain, (1,0,2)), M=M, fburnin=fburnin)

def emcee_ptchain_autocorrelation_lengths(ptchain, M=5, fburnin=None):
    r"""Returns an array of shape ``(Ntemp, Nparams)`` giving the estimated
    autocorrelation lengths for each parameter across each temperature
    of the parallel-tempered set of chains.  If a particular ACL
    cannot be estimated, that element of the array will be ``None``.
    See :func:`emcee_chain_autocorrelation_lengths` for a description
    of the optional arguments.

    """

    if fburnin is None:
        fburnin = _default_burnin(M)

    istart = int(round(fburnin*ptchain.shape[2]))

    return autocorrelation_length_estimate(np.mean(ptchain[:, :, istart:, :], axis=1), axis=1)

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

    if np.any(np.isnan(acls)):
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

    if np.any(np.isnan(acls)):
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

def emcee_gelman_rubin_r(chain, fburnin=None):
    r"""Returns the Gelman-Rubin R convergence statistic applied to
    individual walkers' trajectories in each parameter.

    """

    if fburnin is None:
        fburnin = _default_burnin(5)

    istart = int(round(fburnin*chain.shape[1]))

    chain = chain[:,istart:,:]

    n = chain.shape[1]
    m = chain.shape[0]

    walker_means = np.mean(chain, axis=1)
    walker_variances = np.var(chain, axis=1)

    walker_mean_var = np.var(walker_means, axis=0)
    walker_var_mean = np.mean(walker_variances, axis=0)

    sigma2 = (n - 1.0)/n*walker_var_mean + walker_mean_var

    Vest2 = sigma2 + walker_mean_var / m

    return Vest2 / walker_var_mean

def waic(lnlikes, fburnin=None):
    r"""Returns an estimate of the WAIC from an emcee sampler's lnlike
    (should be of shape ``(nwalkers, nsteps)``).  The WAIC is defined
    by

    .. math::

      \mathrm{WAIC} = -2 \left( \left\langle \ln \mathcal{L} \right\rangle - \mathrm{Var}\, \ln\mathcal{L} \right).

    See Gelman, Hwang, and Vehtari (2013) for a motivation for this
    quantity in terms of an unbiased estimate of the expected log
    pointwise predictive density.

    """

    if fburnin is None:
        fburnin = _default_burnin(5)

    istart = int(round(fburnin*lnlikes.shape[1]))

    lnlikes = lnlikes[:,istart:]

    mu = np.mean(lnlikes)
    v = np.var(lnlikes)

    return -2.0*(mu - v)
