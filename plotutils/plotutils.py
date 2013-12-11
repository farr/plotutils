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

import bounded_kde as bk
import bz2
import gzip
import log_kde as lk
import numpy as np
import matplotlib.pyplot as pp
import os.path 
import scipy.interpolate as si
import scipy.stats as ss
import scipy.stats.mstats as sm

def load_header_data(file, header_commented=False):
    """Load data from the file, using header for column names.

    :param file: A file object or filename.

    :param header_commented: If ``True``, discard the first character
      of header as comment marker."""

    def do_read(file):
        header=file.readline()

        if header_commented:
            header=header[1:].split()
        else:
            header=header.split()
            
        return np.loadtxt(file, dtype=[(h, np.float) for h in header])

    if isinstance(file, str):
        f,ext = os.path.splitext(file)
        
        if ext == '.gz':
            with gzip.open(file, 'r') as inp:
                return do_read(inp)
        elif ext == '.bz2':
            with bz2.BZ2File(file, 'r') as inp:
                return do_read(inp)
        else:
            with open(file, 'r') as inp:
                return do_read(inp)
    else:
        return do_read(inp)

def decorrelated_2d_histogram_pdf(pts, xmin=None, xmax=None, ymin=None, ymax=None):
    """Returns ``(XS, YS, ZS)``, with ``ZS`` of shape ``(Nx,Ny)`` and
    ``XS`` and ``YS`` of shape ``(Nx+1,Ny+1)`` giving the height and
    box corners, respectively, of the histogram estimate of the PDF
    from which ``pts`` are drawn.  The bin widths and orientations are
    chosen optimally for the convergence of the histogram to the true
    PDF under a squared-error metric; the automatic binning is tuned
    to work particularly well for multivariate Gaussians.

    Note: the first index of ZS varies with the X coordinate, while
    the second varies with the y coordinate.  This is consistent with
    :func:`pp.pcolor`, but inconsistent with :func:`pp.imshow` and
    :func:`pp.contour`.

    :param pts: The sample points, of shape ``(Npts,2)``.

    :param xmin: Minimum value in x.  If ``None``, use minimum data
      value.

    :param xmax: Maximum value in x.  If ``None``, use minimum data
      value.

    :param ymin: Minimum value in y.  If ``None``, use minimum data
      value.
    
    :param ymax: Maximum value in y.  If ``None``, use minimum data
      value."""

    if xmin is None:
        xmin = np.min(pts[:,0])
    if xmax is None:
        xmax = np.max(pts[:,0])
    if ymin is None:
        ymin = np.min(pts[:,1])
    if ymax is None:
        ymax = np.max(pts[:,1])

    cov=np.cov(pts, rowvar=0)
    mu=np.mean(pts, axis=0)

    # cov = L*L^T
    d, L = np.linalg.eig(cov)

    rescaled_pts = np.dot(pts-mu, L)
    rescaled_pts = rescaled_pts / np.reshape(np.sqrt(d), (1, 2))

    h = (42.5/pts.shape[0])**0.25

    Nx=int((np.max(rescaled_pts[:,0])-np.min(rescaled_pts[:,0]))/h + 0.5)
    Ny=int((np.max(rescaled_pts[:,1])-np.min(rescaled_pts[:,1]))/h + 0.5)

    H,xs,ys = np.histogram2d(rescaled_pts[:,0], rescaled_pts[:,1], bins=(Nx,Ny))

    # Backwards to account for the ordering in histogram2d.
    YS_RESCALED,XS_RESCALED=np.meshgrid(ys, xs)

    HPTS_RESCALED=np.column_stack((XS_RESCALED.flatten(),
                                   YS_RESCALED.flatten()))

    HPTS=np.dot(HPTS_RESCALED*np.reshape(np.sqrt(d), (1,2)), L.T) + mu

    XS=np.reshape(HPTS[:,0], (Nx+1,Ny+1))
    YS=np.reshape(HPTS[:,1], (Nx+1,Ny+1))

    return XS,YS,H

def interpolated_quantile(sorted_pts, quantile):
    """Returns a linearly interpolated quantile value.

    :param sorted_pts: A sorted array of points.

    :param quantile: The quantile desired."""

    N=sorted_pts.shape[0]

    idx=N*quantile
    lidx=int(np.floor(idx))
    hidx=int(np.ceil(idx))

    return (idx-lidx)*sorted_pts[lidx] + (hidx-idx)*sorted_pts[hidx]

def plot_interval(pts, levels, *args, **kwargs):
    """Plot probability intervals corresponding to ``levels`` in 1D.
    Additional args are passed to :func:`pp.axvline`.  The chosen
    levels are symmetric, in that they have equal probability mass
    outside the interval on each side.

    :param pts: Shape ``(Npts,)`` array of samples.

    :param levels: Sequence of levels to plot."""

    for level in levels:
        low,high = sm.mquantiles(pts, [0.5*(1.0-level), 0.5+0.5*level])
        pp.axvline(low, *args, **kwargs)
        pp.axvline(high, *args, **kwargs)

def plot_greedy_kde_interval_2d(pts, levels, xmin=None, xmax=None, ymin=None, ymax=None, Nx=100, Ny=100, cmap=None, colors=None, *args, **kwargs):
    """Plots the given probability interval contours, using a greedy
    selection algorithm.  Additional arguments passed to
    :func:`pp.contour`.

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

    pp.contour(XS, YS, ZS, zvalues, colors=colors, cmap=cmap, *args, **kwargs)

def plot_greedy_histogram_interval_2d(pts, levels, xmin=None, xmax=None, ymin=None, ymax=None, Nx=100, Ny=100, 
                                      cmap=None, colors=None, *args, **kwargs):
    """Plot probability interval contours estimated from a histogram
    PDF of the given points.  The number of bins in each dimension is
    chosen optimally for minimizing the squared error with a Gaussian
    PDF.  Additional arguments passed to :func:`pp.contour`.

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

    :param Nx: Number of divisions in x for contour resolution.

    :param Ny: Number of divisions in y for contour resolution.

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

    XS,YS,H=decorrelated_2d_histogram_pdf(pts, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    XS_CENTER=0.25*(XS[:-1,:-1]+XS[:-1,1:]+XS[1:,:-1]+XS[1:,1:])
    YS_CENTER=0.25*(YS[:-1,:-1]+YS[:-1,1:]+YS[1:,:-1]+YS[1:,1:])

    pts_post=si.griddata(np.column_stack((XS_CENTER.flatten(),
                                          YS_CENTER.flatten())),
                         H.flatten(),
                         pts,
                         method='linear',
                         fill_value=0.0)

    ipost=np.argsort(pts_post)[::-1]

    post_levels = [pts_post[ipost[int(l*Npts+0.5)]] for l in levels]
        
    pxs=np.linspace(xmin, xmax, Nx)
    pys=np.linspace(ymin, ymax, Ny)

    PXS,PYS=np.meshgrid(pxs, pys)

    posts=si.griddata(np.column_stack((XS_CENTER.flatten(),
                                       YS_CENTER.flatten())),
                      H.flatten(),
                      np.column_stack((PXS.flatten(), PYS.flatten())),
                      method='linear',
                      fill_value=0.0)
    posts=np.reshape(posts, (Nx,Ny))

    pp.contour(PXS, PYS, posts, post_levels, colors=colors, cmap=cmap, origin='lower', extent=(xmin,xmax,ymin,ymax), *args, **kwargs)


def plot_kde_posterior(pts, xmin=None, xmax=None, N=100, periodic=False, low=None, high=None, log=False, *args, **kwargs):
    """Plots the a KDE estimate of the posterior from which ``pts``
    are drawn.  Extra arguments are passed to :func:`pp.plot`.

    :param pts: Shape ``(Npts,)`` array of samples.

    :param xmin: Minimum x value.  If ``None``, will be derived from
      ``pts``.

    :param xmax: Maximum x value.  If ``None``, will be derived from
      ``pts``.

    :param N: Number of intervals across ``(xmin, xmax)`` in plot.

    :param periodic: If true, then the function is periodic on the
      interval.

    :param low: If not ``None``, indicates a lower boundary for the
      domain of the PDF.  

    :param high: If not ``None``, indicates an upper boundary for the
      domain of the PDF."""

    sigma = np.std(pts)

    if xmin is None and not log:
        xmin = np.min(pts)-0.5*sigma
        if low is not None:
            xmin = max(low, xmin)
    if xmax is None and not log:
        xmax = np.max(pts)+0.5*sigma
        if high is not None:
            xmax = min(high, xmax)

    if not log:
        xs=np.linspace(xmin, xmax, N)

    if periodic:
        period=xmax-xmin
        kde=ss.gaussian_kde(pts)
        pp.plot(xs, kde(xs)+kde(xs+period)+kde(xs-period), *args, **kwargs)
    elif low is not None or high is not None:
        kde=bk.Bounded_kde(pts, low=low, high=high)
        pp.plot(xs, kde(xs), *args, **kwargs)
    elif log:
        assert low is None, 'cannot impose low boundary in log-space'
        assert high is None, 'cannot impose high boundary in log-space'

        lpts = np.log(pts)
        sigma = np.std(lpts)

        if xmin is None or xmin < 0:
            xmin = np.min(lpts) - 0.5*sigma
        else:
            xmin = np.log(xmin)

        if xmax is None:
            xmax = np.max(lpts) + 0.5*sigma
        else:
            xmax = np.log(xmax)

        kde=lk.Log_kde(pts)

        xs = np.exp(np.linspace(xmin, xmax, N))

        pp.plot(xs, kde(xs), *args, **kwargs)
        pp.xscale('log')
    else:
        kde=ss.gaussian_kde(pts)
        pp.plot(xs, kde(xs), *args, **kwargs)

def plot_histogram_posterior(pts, xmin=None, xmax=None, log=False, **args):
    """Plots a histogram estimate of the posterior from which ``pts``
    are drawn.  Extra arguments are passed to :func:`pp.hist`.

    :param pts: Shape ``(Npts,)`` array of samples.

    :param xmin: Minimum x value.  If ``None``, will be derived from
      ``pts``.

    :param xmax: Maximum x value.  If ``None``, will be derived from
      ``pts``.

    :param fmt: Line format; see :func:`pp.plot`.

    :param log: If ``True`` compute and plot histogram in log-space.

    """

    if log:
        pts = np.log(pts)

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

    if log:
        pp.hist(np.exp(pts), bins=np.exp(np.linspace(xmin, xmax, Nbins+1)), **args)
    else:
        pp.hist(pts, bins=Nbins, **args)

    # Just because matplotlib sometimes puts the y-axis above 0:
    pp.axis(ymin=0)

    if log:
        pp.xscale('log')

def plot_kde_posterior_2d(pts, xmin=None, xmax=None, ymin=None, ymax=None, Nx=100, Ny=100, cmap=None, log=False):
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

    :param cmap: The colormap, passed to :func:`pp.imshow`.

    :param log: If ``True`` compute and plot the density in log-space.

    """

    if log:
        pts = np.log(pts)

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
    XCENTERS = 0.25*(XS[:-1, :-1] + XS[:-1, 1:] + XS[1:,:-1] + XS[1:, 1:])
    YCENTERS = 0.25*(YS[:-1, :-1] + YS[:-1, 1:] + YS[1:,:-1] + YS[1:, 1:])
    ZS=np.reshape(kde(np.row_stack((XCENTERS.flatten(), YCENTERS.flatten()))), (Nx-1, Ny-1))

    if log:
        XS = np.exp(XS)
        YS = np.exp(YS)

        xmin = np.exp(xmin)
        xmax = np.exp(xmax)
        ymin = np.exp(ymin)
        ymax = np.exp(ymax)

    pp.pcolormesh(XS, YS, ZS, cmap=cmap)
    pp.axis(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    if log:
        pp.xscale('log')
        pp.yscale('log')

def plot_histogram_posterior_2d(pts, log=False, cmap=None):
    """Plots a 2D histogram density of the given points.

    :param pts: An ``(Npts, 2)`` array of points.

    :param log: If ``True``, then compute and plot the histogram in
      log-space.

    :param cmap: Passed to :func:`pp.imshow` as colormap.

    """

    if log:
        XS,YS,HS=decorrelated_2d_histogram_pdf(np.log(pts))
        XS = np.exp(XS)
        YS = np.exp(YS)
    else:
        XS,YS,HS=decorrelated_2d_histogram_pdf(pts)

    xmin=np.min(XS.flatten())
    xmax=np.max(XS.flatten())
    ymin=np.min(YS.flatten())
    ymax=np.max(YS.flatten())

    # Plot a zero-level background...
    pp.pcolormesh(np.array([[xmin, xmax], [xmin, xmax]]),
                  np.array([[ymin, ymin], [ymax, ymax]]),
                  np.array([[0.0]]), cmap=cmap)

    pp.pcolormesh(XS,YS,HS, cmap=cmap)

    pp.axis(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    if log:
        pp.xscale('log')
        pp.yscale('log')

def plot_cumulative_distribution(pts, *args, **kwargs):
    """Plots the 1D normalized empirical CDF for the given points.
    Additional arguments are passed to matplotlib's ``plot``.

    """
    pts = np.atleast_1d(pts)
    pts = np.sort(np.concatenate(([0], pts)))

    pp.plot(pts, np.linspace(0, 1, pts.shape[0]), *args, **kwargs)


    
