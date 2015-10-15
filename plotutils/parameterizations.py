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
    r"""Returns the log of the Jacobian factor,

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

    return p

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

def increasing_values(p):
    r"""Returns the values for parameters ``p`` which are constrained to be
    always increasing.

    The parameterisation is 

    .. math::

      p_i = \begin{cases}
        x_0 & i = 0 \\
        \log\left( x_i - x_{i-1} \right) & \mathrm{otherwise}
      \end{cases}

    Note that :math:`-\infty < p < \infty`.

    """

    p = np.atleast_1d(p)
    x = np.zeros(p.shape)

    x[0] = p[0]
    for i in range(1, x.shape[0]):
        x[i] = x[i-1] + np.exp(p[i])

    return x

def increasing_params(x):
    """Returns the parameters associated with the values ``x`` which
    should be increasing.  

    """

    x = np.atleast_1d(x)
    x = np.sort(x)
    p = np.zeros(x.shape)

    p[0] = x[0]
    for i in range(1, x.shape[0]):
        p[i] = np.log(x[i] - x[i-1])

    return p

def increasing_log_jacobian(p):
    r"""Returns the log of the Jacobian factor

    .. math::

      \left| \frac{\partial x}{\partial p} \right|

    for the parameters p.

    """

    return np.sum(p[1:])

def _logitab(x, a, b):
    return np.log(x - a) - np.log(b - x)

def _invlogitab(y, a, b):
    if y > 0.0:
        ey = np.exp(-y)
        return (a*ey + b)/(1.0 + ey)
    else:
        ey = np.exp(y)
        return (b*ey + a)/(1.0 + ey)

e^y(b-a)/(e^y+1)^2
    
def _logitablogjac(y, a, b):
    if y < 0.0:
        ey = np.exp(y)
        return np.log(b-a) + y - 2.0*np.log1p(ey)
    else:
        ey = np.exp(-y)
        return np.log(b-a) - y - 2.0*np.log1p(ey) 

def _stable_polynomial_roots_logjac(p, rmin, rmax):
    """

    """
    p = np.atleast_1d(p)

    a = -(rmax-rmin)
    b = rmax

    n = p.shape[0]

    rs = []
    lj = 0.0
    for i in range(0, n-1, 2):
        y = _invlogitab(p[i], a, b)
        lj += _logitablogjac(p[i], a, b)

        if y > 0.0:
            # Complex root
            x = _invlogitab(p[i+1], -rmax, -rmin)
            lj += _logitablogjac(p[i+1], -rmax, -rmin)

            rs.append(x + y*1j)
            rs.append(x - y*1j)
            b = y
        else:
            rs.append(y-rmin)
            b = y

            x = _invlogitab(p[i+1], a, b)
            lj += _logitablogjac(p[i+1], a, b)
            rs.append(x-rmin)
            b = x

    if n % 2 == 1:
        if b > 0.0:
            b = 0.0 # The final root must be negative real, no matter what
        x = _invlogitab(p[n-1], a, b)
        lj += _logitablogjac(p[n-1], a, b)

        rs.append(x - rmin)

    return np.array(rs, dtype=np.complex), lj

def stable_polynomial_roots(p, rmin, rmax):
    r"""A parameterisation of the roots of a real, 'stable' polynomial.

    A stable polynomial has roots with a negative real part; it is the
    characteristic polynomial for a linear ODE with decaying
    solutions.  The parameterisation provides a mapping from
    :math:`\mathbb{R}^n` to the roots that is one-to-one; there are no
    root-permutation degeneracies.  The log-Jacobian function produces
    a flat distribution on the real and imaginary (if any) parts of
    the roots.

    :param p: The array giving the parameters in :math:`\mathbb{R}^n`
      for the roots of the polynomial.

    :param rmin: The real part of all the roots is bounded below
      :math:`-r_\mathrm{min}`.

    :param rmax: The real part of all the roots is bounded above
      :math:`-r_\mathrm{max}`, and the imaginary parts of all the
      roots are bounded between :math:`-r_\mathrm{max}` and
      :math:`r_\mathrm{max}`.

    The parameterisation uses a 'bounded logit' transformation to map
    ranges of reals to :math:`\pm \infty`:

    .. math::

      \mathrm{logit}\left(x; a, b\right) = \log(x-a) - log(b-x)

    The mapping of the roots proceeds as follows.  First, discard the
    roots with strictly negative imaginary parts (these are the
    conjugates of corresponding roots with strictly positive imaginary
    parts, so we lose no information).  Then, sort the remaining roots
    in order of decreasing imaginary part; if there are any strictly
    real roots, with imaginary part zero, sort these in decreasing
    order, from least negative to most negative.  

    Let :math:`a = -\left( r_\mathrm{max} - r_\mathrm{min} \right)`
    and :math:`b = r_\mathrm{max}`.  Then, proceeding by pairs of
    roots:

      * If the imaginary part of root ``i`` is greater than zero, then 

        .. math::

          p_i = \mathrm{logit}\left(\mathrm{im}\left(r_i\right); a, b\right) \\
          p_{i+1} = \mathrm{logit}\left(\mathrm{re}\left(r_i\right); -r_\mathrm{max}, -r_\mathrm{min} \right) 

        Then set :math:`b = \mathrm{im} r_i`, and proceed to the next pair.

      * If the imaginary part of root ``i`` is zero, then 

        .. math::

          p_i = \mathrm{logit}\left( r_i + r_\mathrm{min}; a, b \right) \\
          p_{i+1} = \mathrm{logit}\left( r_{i+1} + r_\mathrm{min}; a, r_i + r_\mathrm{min} \right)

        then set :math:`b = r_{i+1} + r_\mathrm{min}`.

      * If the number of roots is odd, then the final root must be
        real, and is the smallest of all the real roots:

        .. math::

          p_{N-1} = \mathrm{logit}\left( r_{N-1} + r_\mathrm{min}; a, \mathrm{min}(b, 0) \right).

    Intuitively, you can imagine constructing the parameterisation
    using a line that sweeps down the imaginary axis, starting from
    :math:`i r_\mathrm{max}`; as it hits each complex root, it records
    the logit of that root's imaginary and real parts (which must lie
    within certain bounds), and then sets the maximum bound for the
    imaginary part of the next root.  Once all the complex roots have
    been parameterised, a line begins at :math:`-r_\mathrm{min}` on
    the real axis and sweeps left; as it hits each real root, it
    records the logit of that root between the current bounds, and
    resets the bound on the maximum value of the next root to the
    current real root.

    The reverse transformation begins by unpacking the first parameter
    value using the inverse logit transform; if this value is
    positive, then the root is complex, and the next parameter
    correponds to the real part.  If the value is negative, then this
    a real root, and the following roots are also real (and negative).
    Bounds on the subsequent root values are set accordingly in either
    case for the next inverse logit transformation.  

    The parameterisation maps the allowed, sorted root space onto the
    entire :math:`\mathbb{R}^N` real space in a one-to-one way.

    """

    return _stable_polynomial_roots_logjac(p, rmin, rmax)[0]

def stable_polynomial_log_jacobian(p, rmin, rmax):
    return _stable_polynomial_roots_logjac(p, rmin, rmax)[1]

def stable_polynomial_params(r, rmin, rmax):
    r = np.atleast_1d(r)
    n = r.shape[0]

    cplx_r = r[np.imag(r) > 0.0]
    real_r = np.real(r[np.imag(r) == 0.0])

    cplx_r = cplx_r[np.argsort(np.imag(cplx_r))][::-1] # Decreasing imag
    real_r = real_r[np.argsort(real_r)][::-1] # Decreasing real

    a = -(rmax-rmin)
    b = rmax

    p = []
    for rc in cplx_r:
        y = np.imag(rc)
        p.append(_logitab(y, a, b))

        x = np.real(rc)
        p.append(_logitab(x, -rmax, -rmin))

        b = y

    if n % 2 == 1:
        last_r = real_r[-1]
        real_r = real_r[:-1]
        
    for rr in real_r:
        y = rr + rmin
        p.append(_logitab(y, a, b))
        b = y

    if n % 2 == 1:
        if b > 0.0:
            b = 0.0
        y = last_r + rmin
        p.append(_logitab(y, a, b))
        
    return np.array(p)
