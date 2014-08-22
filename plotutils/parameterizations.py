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
