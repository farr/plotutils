# contour.py: Class for computing contour levels and membership.
# Copyright (C) 2013 Will M. Farr <w-farr@northwestern.edu>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see
# <http://www.gnu.org/licenses/>.

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
