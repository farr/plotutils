"""Useful parameterizations for various commonly-found elements of
models.

"""

import numpy as np

def _cov_matrix_dim(x):
    x = np.atleast_1d(x)
    M = x.shape[0]
    N = int(round(0.5*(np.sqrt(1.0 + 8.0*M) - 1)))

    if not ((N+1)*N/2 == M):
        raise ValueError('input must have shape (N+1)*N/2 for some N')
        
    return N
    
def _cov_params_to_cholesky(x):
    N = _cov_matrix_dim(x)

    y = np.zeros((N,N))
    y[np.tril_indices(N)] = x
    y.flat[::(N+1)] = np.exp(y.flat[::(N+1)])

    return y

def cov_matrix(x):
    r"""Returns a covariance matrix from the parameters ``x``, which should
    be of shape ``((N+1)*N/2,)`` for an ``(N,N)`` covariance matrix.

    The parametersation is taken from the `Stan
    <http://mc-stan.org/>`_ sampler, and is as follows.  The
    parameters ``x`` are the lower-triangluar elements of a matrix,
    :math:`y`.  Let 

    .. math::

      z_{ij} = \begin{cases}
        \exp\left(y_{ij}\right) & i = j \\
        y_{ij} & \mathrm{otherwise}
      \end{cases}

    Then the covariance matrix is 

    .. math::

      \Sigma = z z^T

    With this parameterization, there are no constraints on the
    components of ``x``.

    """
    y = _cov_params_to_cholesky(x)

    return np.dot(y, y.T)

def cov_parameters(m):
    """Given a covariance matrix, ``m``, returns the parameters associated
    with it through the :func:`cov_matrix` function.

    """
    m = np.atleast_2d(m)
    N = m.shape[0]

    z = np.linalg.cholesky(m)
    z.flat[::(N+1)] = np.log(z.flat[::(N+1)])

    return z[np.tril_indices(N)]

def cov_log_jacobian(x):
    r"""Returns the log of the determinant of the Jacobian of the
    transformation that produces the covariance matrix from parameters
    ``x``:

    .. math::
    
      \log |J| = \log \left| \frac{\partial \Sigma}{\partial x} \right|

    """

    N = _cov_matrix_dim(x)
    y = _cov_params_to_cholesky(x)

    expts = N - np.arange(1, N+1) + 2

    return N*np.log(2) + np.dot(np.log(np.diag(y)), expts)

def _logit(x):
    return np.log(x) - np.log1p(-x)

def _invlogit(y):
    return 1.0/(1.0 + np.exp(-y))

def _usimp_zs(p):
    p = np.atleast_1d(p)
    N = p.shape[0]+1
    ks = N - np.arange(1, N)
    
    zs = _invlogit(p - np.log(ks))

    return zs

def usimp_lengths(p):
    """Given ``N-1`` parameters, ``p``, returns ``N`` positive values that
    sum to one.  The transformation comes from the Stan manual.
    Imagine a stick that begins with unit length; the parameters are
    logit-transformed fractions of the amount of the stick remaining
    that is broken off in each of ``N-1`` steps to produce the ``N``
    lengths.

    """
    p = np.atleast_1d(p)
    N = p.shape[0]+1

    zs = _usimp_zs(p)

    xs = np.zeros(N)
    for i in range(xs.shape[0]-1):
        xs[i] = zs[i]*(1 - np.sum(xs[:i]))
    xs[N-1] = 1 - np.sum(xs[:N-1])

    return xs

def usimp_parameters(x):
    """Returns the ``N-1`` unit simplex parameters that will produce the
    ``N`` lengths ``x``.

    """
    x = np.atleast_1d(x)
    N = x.shape[0]
    zs = np.zeros(N-1)
    csxs = np.cumsum(x)

    for i in range(N-1):
        zs[i] = x[i]/(1-csxs[i]+x[i])

    ks = N - np.arange(1, N)

    ys = _logit(zs) + np.log(ks)

    return ys

def usimp_log_jacobian(p):
    """Returns the log of the Jacobian factor, 

    .. math::

      \left| \frac{\partial x}{\partial p} \right|

    where :math:`x` are the unit simplex lengths.

    """
    p = np.atleast_1d(p)
    N = p.shape[0]
    zs = _usimp_zs(p)

    xs = usimp_lengths(p)
    csxs = np.cumsum(xs) - xs # Starts from zero

    log_j_terms = np.log(zs) + np.log1p(-zs) + np.log1p(-csxs[:-1])

    return np.sum(log_j_terms)

def bounded_values(p, low=np.NINF, high=np.inf):
    r"""Returns the values, each bounded between ``low`` and ``high`` (one
    of these can be ``None``) associated with the parameters ``p``.

    The parameterisation is 

    .. math::

      p = \log\left( x - \mathrm{low} \right) - \log\left( \mathrm{high} - x \right)

    if both lower and upper limits are given, and 

    .. math::

      p = \log\left( x - \mathrm{low} \right)

    or 

    .. math::

      p = -\log\left( \mathrm{high} - x \right)

    if only one limit is given.

    :param p: The parameters associated with the values.
      :math:`-\infty < p < \infty`

    :param low: The lower bound on the parameters.  Can be a vector
      that matches the shape of ``p``.

    :param high: The upper bound on the parameters.  Can be a vector
      that matches the shape of ``p``.

    """

    p = np.atleast_1d(p)
    x = np.zeros(p.shape)

    for i, (p, l, h) in enumerate(np.broadcast(p, low, high)):
        if l == np.NINF:
            if h == np.inf:
                raise ValueError('bounded_values: must supply at least one limit')
            else:
                # Only upper limit
                x[i] = h - np.exp(-p)
        else:
            if h == np.inf:
                # Only lower limit.
                x[i] = l + np.exp(p)
            else:
                # Both bounds
                ep = np.exp(p)
                x[i] = (ep*h + l)/(ep + 1)

    return x

def bounded_params(x, low=np.NINF, high=np.inf):
    """Returns the parameters associated with the values ``x`` that are
    bounded between ``low`` and ``high``.

    """

    x = np.atleast_1d(x)
    p = np.zeros(x.shape)

    for i, (x, l, h) in enumerate(np.broadcast(x, low, high)):
        if l == np.NINF:
            if h == np.inf:
                raise ValueError('bounded_params: must supply at least one limit')
            else:
                # Only upper limit
                p[i] = -np.log(h - x)
        else:
            if h == np.inf:
                # Only lower limit
                p[i] = np.log(x - l)
            else:
                p[i] = np.log(x - l) - np.log(h - x)

def bounded_log_jacobian(p, low=np.NINF, high=np.inf):
    r"""Returns the log of the Jacobian factor 

    .. math::

      \left| \frac{\partial x}{\partial p} \right|

    for the bounded parameters p.

    """

    lj = 0.0

    for p, l, h in np.broadcast(p, low, high):
        if l == np.NINF:
            if h == np.inf:
                raise ValueError('bounded_log_jacobian: must supply at least one limit')
            else:
                # Only upper limit
                lj -= p
        else:
            if h == np.inf:
                # Only lower limit
                lj += p
            else:
                # Both limits
                lj += np.log(h-l) + p - 2.0*np.log1p(np.exp(p))

    return lj
