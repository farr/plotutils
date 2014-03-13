import numpy as np
from scipy.stats import gaussian_kde

class Contour(object):
    """A contour calculator.  Takes as input a set of points drawn from
    some distribution, and a set of credible levels.  Then
    :meth:`Contour.in_contours` computes for any set of points which
    ones like within the given credible contours and which are
    outside.

    """

    def __init__(self, data, contours):
        """Initialize the contours object. 

        :param data: A ``(Npts, Ndim)`` array of points drawn from the
          distribution in question.

        :param contours: A sequence of contours to compute for the
        distribution in question.

        """

        N = data.shape[0]
        self._kde = gaussian_kde(data.T)
        self._contours = list(contours)
        ps = np.sort(self.kde(data.T))[::-1]
        

        self._contour_limits = []
        for c in self.contours:
            ic = int(round(c*N))
            if ic < 0:
                ic = 0
            if ic >= N:
                ic = N-1

            self.contour_limits.append(ps[ic])

    @property
    def kde(self):
        """A Gaussian kernel density estimator for the stored data set.

        """

        return self._kde

    @property
    def contours(self):
        """Returns the quantiles of the contours represented by this object.

        """
        return self._contours

    @property
    def contour_limits(self):
        """Returns the lower bound on the posterior estimate for inclusion in
        each of the contours.

        """
        return self._contour_limits

    def in_contours(self, pts):
        """Returns arrays indicating whether the given points are within each
        of the contours of the object.

        :param pts: A ``(Npts, Ndim)`` array of points to test against
          each contour.

        :return: A sequence of the same shape as
          :data:`Contours.contours` containing boolean arrays
          indicating which points lie within the contour.

        """

        ps = self.kde(pts.T)

        in_arrays = []
        for cl in self.contour_limits:
            in_arrays.append(ps > cl)

        return in_arrays
