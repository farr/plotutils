import numpy as np
from scipy.stats import gaussian_kde

class Log_kde(gaussian_kde):
    r"""Represents a one-dimensional Gaussian kernel density estimator
    conducted in log-space, according to the formula

    .. math::
    
      \frac{dN}{d\ln x}(\ln x) = x \frac{dN}{dx}(x)

    The KDE occurs on the left-hand-side, but the reported
    distribution is :math:`dN/dx(x)`.
    """

    def __init__(self, pts, *args, **kwargs):
        """Initialize the KDE with the given points.  Extra arguments
        are passed to :class:`gaussian_kde`.

        :param pts: A one-dimensional array of sample points whose
          density is to be estimated.
        """

        assert pts.ndim == 1, 'must have one-dimensional points'
        assert np.all(pts > 0), 'must have positive points'

        super(Log_kde, self).__init__(np.log(pts), *args, **kwargs)

        
    def evaluate(self, xs):
        """Returns an estimate of the density at the given points."""

        xs = np.atleast_1d(xs)
        assert xs.ndim == 1, 'must have one-dimensional points'
        assert np.all(xs>0), 'must have positive points'

        pdf = super(Log_kde, self).evaluate(np.log(xs))

        return pdf / xs

    __call__ = evaluate
