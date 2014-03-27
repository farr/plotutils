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

def autocorrelation_function(series, axis=0):
    """Returns the autocorrelation function of the given series.  The
    function is normalised so that it is 1 at zero lag.

    """
    series = np.atleast_1d(series)
    shape = np.array(series.shape)
    m = [slice(None)] * len(shape)

    n = _next_power_of_two(shape[axis]*2)
    m[axis] = slice(0, shape[axis])
    shape[axis] = n

    padded_series = np.zeros(shape)
    padded_series[m] = series - np.mean(series, axis=axis)

    ps_tilde = np.fft.fft(padded_series, axis=axis)

    acf = np.real(np.fft.ifft(ps_tilde*np.conj(ps_tilde), axis=axis))[m]
    m[axis] = 0
    acf /= acf[m]

    return acf

def autocorrelation_length_estimate(series, acf=None, M=5, axis=0):
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
        acf = autocorrelation_function(series, axis=axis)

    summed_acf = np.cumsum(np.abs(acf), axis=axis) + 1.0 # Don't forget about
                                                         # double the zero-lag
                                                         # component
    m = [slice(None)] * len(acf.shape)
    m[axis] = slice(0,10)
    acls = (np.cumsum(np.ones(summed_acf.shape), axis=axis)-1.0)/M
    diffs = summed_acf - acls
    if np.any(diffs < 0):
        i = np.argmin(np.abs(diffs), axis=axis)
        j = tuple(np.indices(i.shape))
        return acls[j[:axis] + (i,) + j[axis:]]
    else:
        return None
