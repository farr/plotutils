# plotutils.py: Useful functions for plotting PDFs using matplotlib.
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
import matplotlib.pyplot as pp
import scipy.stats as ss

def load_header_data(file, header_commented=False):
    """Load data from the file, using header for column names.

    :param file: A file object.

    :param header_commented: If ``True``, discard the first character
      of header as comment marker."""

    header=file.readline()

    if header_commented:
        header=header[1:].split()
    else:
        header=header.split()

    return np.loadtxt(file, dtype=[(h, np.float) for h in header])

def bin_numbers_2d(pts, xmin=None, xmax=None, ymin=None, ymax=None):
    """Returns ``(Nx,Ny)`` the number of bins along each dimension
    chosen according to optimality criterion for minimizing the
    squared error between histogram and true Gaussian PDF in 2D.

    :param pts: Shape ``(Npts, 2)`` array of sample points.

    :param xmin: Minimum dimension in x.  If ``None``, obtained from
      ``pts``.

    :param xmax: Maximum dimension in x.  If ``None``, obtained from
      ``pts``.

    :param ymin: Minimum dimension in y.  If ``None``, obtained from
      ``pts``.

    :param ymax: Maximum dimension in y.  If ``None``, obtained from
      ``pts``."""
    
    N = pts.shape[0]

    if xmin is None:
        xmin = np.min(pts[:,0])
    if xmax is None:
        xmax = np.max(pts[:,0])
    if ymin is None:
        ymin = np.min(pts[:,1])
    if ymax is None:
        ymax = np.max(pts[:,1])

    # Covariance matrix
    sigma = np.cov(pts, rowvar=0)
    
    # Principal directions and widths
    evals,evecs=np.linalg.eig(sigma)

    # Scaled according to optimal width for Gaussian PDF in 2D
    evals *= np.sqrt(42.5/N)

    # Rotated back to original frame
    width_matrix = np.dot(evecs, np.dot(np.diag(evals), evecs))

    # Projected onto original x-y axes
    hx = np.abs(width_matrix[0,0])
    hy = np.abs(width_matrix[1,1])

    return int((xmax-xmin)/hx + 0.5), int((ymax-ymin)/hy + 0.5)

def interpolated_quantile(sorted_pts, quantile):
    """Returns a linearly interpolated quantile value.

    :param sorted_pts: A sorted array of points.

    :param quantile: The quantile desired."""

    N=sorted_pts.shape[0]

    idx=N*quantile
    lidx=int(np.floor(idx))
    hidx=int(np.ceil(idx))

    return (idx-lidx)*sorted_pts[lidx] + (hidx-idx)*sorted_pts[hidx]

def plot_interval(pts, levels, **args):
    """Plot probability intervals corresponding to ``levels`` in 1D.
    Additional args are passed to :function:`pp.axvline`.

    :param pts: Shape ``(Npts,)`` array of samples.

    :params levels: Sequence of levels to plot."""

    Npts=pts.shape[0]
    spts=np.sort(pts)

    for level in levels:
        alpha=0.5*(1.0-level)

        pp.axvline(interpolated_quantile(spts, alpha), **args)
        pp.axvline(interpolated_quantile(spts, 1.0-alpha), **args)
                   

def plot_greedy_kde_interval_2d(pts, levels, xmin=None, xmax=None, ymin=None, ymax=None, Nx=100, Ny=100, cmap=None, colors=None):
    """Plots the given probability interval contours, using a greedy
    selection algorithm.

    :param pts: Array of shape ``(Npts, 2)`` that contains the points
      in question.

    :param levels: Sequence of levels (between 0 and 1) of probability
      intervals to plot.

    :param xmin: Minimum value in x.  If ``None``, use minimum data
      value.

    :param xmax: Maximum value in x.  If ``None``, use minimum data
      value.

    :param ymin: Minimum value in y.  If ``None``, use minimum data
      value.
    
    :param ymax: Maximum value in y.  If ``None``, use minimum data
      value.

    :param Nx: Number of subdivisions in x for contour plot.  (Default
      100.)

    :param Ny: Number of subdivisions in y for contour plot.  (Default
      100.)

    :param cmap: See :func:`pp.contour`.

    :param colors: See :func:`pp.contour`.
      """

    kde=ss.gaussian_kde(pts.T)
    den=kde(pts.T)
    densort=np.sort(den)[::-1]
    Npts=pts.shape[0]

    if xmin is None:
        xmin = np.min(pts[:,0])
    if xmax is None:
        xmax = np.max(pts[:,0])
    if ymin is None:
        ymin = np.min(pts[:,1])
    if ymax is None:
        ymax = np.max(pts[:,1])

    xs = np.linspace(xmin, xmax, Nx)
    ys = np.linspace(ymin, ymax, Ny)

    XS,YS=np.meshgrid(xs,ys)
    ZS=np.reshape(kde(np.row_stack((XS.flatten(), YS.flatten()))), (Nx, Ny))

    zvalues=[]
    for level in levels:
        ilevel = int(Npts*level + 0.5)
        if ilevel >= Npts:
            ilevel = Npts-1
        zvalues.append(densort[ilevel])

    pp.contour(XS, YS, ZS, zvalues, colors=colors, cmap=cmap)

def plot_greedy_histogram_interval_2d(pts, levels, xmin=None, xmax=None, ymin=None, ymax=None, cmap=None, colors=None):
    """Plot probability interval contours estimated from a histogram
    PDF of the given points.  The number of bins in each dimension is
    chosen optimally for minimizing the squared error with a Gaussian
    PDF.

    :param pts: Shape ``(Npts, 2)`` array of samples.

    :param levels: The probability interval levels to plot.

    :param xmin: Minimum value in x.  If ``None``, use minimum data
      value.

    :param xmax: Maximum value in x.  If ``None``, use minimum data
      value.

    :param ymin: Minimum value in y.  If ``None``, use minimum data
      value.
    
    :param ymax: Maximum value in y.  If ``None``, use minimum data
      value.

    :param cmap: See :func:`pp.contour`.

    :param colors: See :func:`pp.contour`."""

    if xmin is None:
        xmin = np.min(pts[:,0])
    if xmax is None:
        xmax = np.max(pts[:,0])
    if ymin is None:
        ymin = np.min(pts[:,1])
    if ymax is None:
        ymax = np.max(pts[:,1])

    Npts = pts.shape[0]

    Nx,Ny=bin_numbers_2d(pts, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    H,xbins,ybins=np.histogram2d(pts[:,0], pts[:,1], bins=(Nx,Ny), range=((xmin,xmax), (ymin,ymax)))

    # histogram2d returns the x samples down the 0-axis, while we want
    # x constant on the 0-axis
    H = np.transpose(H)

    # Bin centers, not bounds.
    xbins = 0.5*(xbins[1:]+xbins[:-1])
    ybins = 0.5*(ybins[1:]+ybins[:-1])

    Hsorted=np.sort(H.flatten())[::-1]
    Hsorted_cum = np.cumsum(Hsorted)

    Hlevels=[]
    for level in levels:
        cum_pts = int(Npts*level + 0.5)
        if cum_pts >= Npts:
            cum_pts = Npts-1

        Hlevels.append(Hsorted[np.nonzero(Hsorted_cum >= cum_pts)[0][0]])

    pp.contour(xbins, ybins, H, colors=colors, cmap=cmap, levels=Hlevels, extent=(xmin,xmax,ymin,ymax))

def plot_kde_posterior(pts, xmin=None, xmax=None, N=100, **args):
    """Plots the a KDE estimate of the posterior from which ``pts``
    are drawn.  Extra keyword arguments are passed to
    :function:`pp.plot`.

    :param pts: Shape ``(Npts,)`` array of samples.

    :param xmin: Minimum x value.  If ``None``, will be derived from
      ``pts``.

    :param xmax: Maximum x value.  If ``None``, will be derived from
      ``pts``.

    :param N: Number of intervals across ``(xmin, xmax)`` in plot."""

    if xmin is None:
        xmin = np.min(pts)
    if xmax is None:
        xmax = np.max(pts)

    kde=ss.gaussian_kde(pts)
    xs=np.linspace(xmin, xmax, N)

    pp.plot(xs, kde(xs), **args)

def plot_histogram_posterior(pts, xmin=None, xmax=None, **args):
    """Plots a histogram estimate of the posterior from which ``pts``
    are drawn.  Extra arguments are passed to :function:`pp.hist`.

    :param pts: Shape ``(Npts,)`` array of samples.

    :param xmin: Minimum x value.  If ``None``, will be derived from
      ``pts``.

    :param xmax: Maximum x value.  If ``None``, will be derived from
      ``pts``.
    
    :param fmt: Line format; see :function:`pp.plot`."""

    if xmin is None:
        xmin=np.min(pts)
    if xmax is None:
        xmax=np.max(pts)

    Npts=pts.shape[0]
    spts=np.sort(pts)

    iqr = spts[int(Npts*0.75+0.5)] - spts[int(Npts*0.25 + 0.5)]

    # Optimal for Gaussian PDF in 1D; minimizes the squared error
    # between true and histogram PDF.
    h = iqr/1.35*(42.5/Npts)**(1.0/3.0)

    Nbins = int((xmax-xmin)/h + 0.5)

    pp.hist(pts, bins=Nbins, **args)

    # Just because matplotlib sometimes puts the y-axis above 0:
    pp.axis(ymin=0)

def plot_kde_posterior_2d(pts, xmin=None, xmax=None, ymin=None, ymax=None, Nx=100, Ny=100, cmap=None):
    """Plot a 2D KDE estimated posterior.

    :param pts: A ``(Npts, 2)`` array of points.

    :param xmin: Minimum x-coordinate.  If ``None``, use the minimum
      value from ``pts``.

    :param xmax: Maximum x-coordinate.  If ``None``, use the maximum
      value from ``pts``.

    :param ymin: Minimum y-coordinate.  If ``None``, use the minimum
      value from ``pts``.

    :param ymax: Maximum y-coordinate.  If ``None``, use the maximum
      value from ``pts``.

    :param Nx: The number of pixels in the x direction.

    :param Ny: The number of pixels in the y direction.

    :param cmap: The colormap, passed to :function:`pp.imshow`."""

    if xmin is None:
        xmin = np.min(pts[:,0])
    if xmax is None:
        xmax = np.max(pts[:,0])
    if ymin is None:
        ymin = np.min(pts[:,1])
    if ymax is None:
        ymax = np.max(pts[:,1])

    kde=ss.gaussian_kde(pts.T)
    XS,YS=np.meshgrid(np.linspace(xmin, xmax, Nx),
                      np.linspace(ymin, ymax, Ny))
    ZS=np.reshape(kde(np.row_stack((XS.flatten(), YS.flatten()))), (Nx, Ny))

    pp.imshow(ZS, origin='lower', cmap=cmap, extent=(xmin, xmax, ymin, ymax), aspect='auto')

def plot_histogram_posterior_2d(pts, cmap=None, xmin=None, xmax=None, ymin=None, ymax=None):
    """Plots a 2D histogram density of the given points.

    :param pts: An ``(Npts, 2)`` array of points.

    :param cmap: Passed to :function:`pp.imshow` as colormap.

    :param xmin: Minimum bound in x.  If ``None`` derived from
      ``pts``.

    :param xmax: Maximum bound in x.  If ``None`` derived from
      ``pts``.

    :param ymin: Minimum bound in y.  If ``None`` derived from
      ``pts``.

    :param ymax: Maximum bound in y.  If ``None`` derived from
      ``pts``."""

    if xmin is None:
        xmin = np.min(pts[:,0])
    if xmax is None:
        xmax = np.max(pts[:,0])
    if ymin is None:
        ymin = np.min(pts[:,1])
    if ymax is None:
        ymax = np.max(pts[:,1])

    Nx,Ny = bin_numbers_2d(pts, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    pp.hexbin(pts[:,0], pts[:,1], gridsize=(Nx,Ny), cmap=cmap, extent=(xmin,xmax,ymin,ymax))
