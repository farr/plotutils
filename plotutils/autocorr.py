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
    series.  The autocorrelation length is defined as the smallest
    integer, :math:`i`, such that 

    .. math::

      \sum_{j < M i} \left| \rho(j) \right| < i

    This empirical relation is intended to estimate in a robust way
    the exponential decay constant for an ACF that decays like
    :math:`\exp(t/\tau)`.  The constant :math:`M` controls how many
    estimated ACLs are included in the calculation of the ACL.

    Returns ``None`` if no such index is present; this indicates that
    the ACL estimate is not converged.

    """
    if acf is None:
        acf = autocorrelation_function(series)

    summed_acf = np.cumsum(np.abs(acf))
    acls = np.arange(0, summed_acf.shape[0])/5.0
    
    acl_selector = summed_acf < acls

    if np.any(acl_selector):
        return acls[np.nonzero(acl_selector)[0][0]]
    else:
        return None
