"""Module containing functions to compute the autocorrelation function
and estimate the associated autocorrelation length of series.

The estimate of the autocorrelation function is based on the method
described at http://www.math.nyu.edu/faculty/goodman/software/acor/
and implemented in the associated ``acor`` C++ code, though the
implementation details differ.

"""

import numpy as np

def _next_power_of_two(i):
    pt = 2
    while pt < i:
        pt *= 2

    return pt

def autocorrelation_function(series):
    """Returns the autocorrelation function of the given series.  The
    function is normalised so that it is 1 at zero lag.

    """
    series = np.atleast_1d(series)
    
    n = _next_power_of_two(series.shape[0]*2)

    padded_series = np.zeros(n)
    padded_series[:series.shape[0]] = series - np.mean(series)

    ps_tilde = np.fft.fft(padded_series)

    acf = np.real(np.fft.ifft(ps_tilde*np.conj(ps_tilde)))
    acf /= acf[0]

    return acf[:series.shape[0]]

def autocorrelation_length_estimate(series, acf=None, M=5):
    r"""Returns an estimate of the autocorrelation length of the given
    series:

    .. math::

      L = \int_{-\infty}^\infty \rho(t) dt

    The estimate is the smallest :math:`L` such that 

    .. math::

      L = 1 + 2*\sum_{j = 1}^{M L} \left| \rho(j) \right|

    In words: the ACL is estimated over a window that is at least
    :math:`M` ACLs long.

    Returns ``None`` if there is no such estimate possible (because
    the series is too short to fit 5 ACLs).

    """
    if acf is None:
        acf = autocorrelation_function(series)

    acl_ests = 2.0*np.cumsum(np.abs(acf)) - 1.0
    sel = M*acl_ests < np.arange(0, acf.shape[0])

    if np.any(sel):
        return acl_ests[np.nonzero(sel)[0][0]]
    else:
        return None

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
