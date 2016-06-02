import bisect
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

def plot_emcee_chains(chain, truths=None, mean=True, fburnin=0):
    """Produces a chain plot of the mean values of each variable at each
    step.  The chain should have the shape ``(nwalkers, nsteps,
    nvars)``, and the resulting grid of plots will be as close to
    square as possible, showing the mean walker position at each step.

    :param chain: An array of shape ``(nwalkers, nsteps, nvars)``
      giving the history of the chain.

    :param truths: Either ``None`` or an iterable giving the truth
      values for each of the parameters, which will be plotted as a
      horizontal line.

    :param mean: If ``True`` (default) plot only the mean of the
      walker ensemble.  Otherwise, plot the evolution of each walker
      in the chain.

    :param fburnin: The fraction of points to discard at the beginning
      of the chain.

    """
    nk = chain.shape[2]
    n = int(np.ceil(np.sqrt(nk)))

    istart = int(round(fburnin*chain.shape[1]))
    
    for k in range(nk):
        pp.subplot(n,n,k+1)

        if mean:
            pp.plot(np.mean(chain[:,istart:,k], axis=0))
        else:
            pp.plot(chain[:,istart:,k].T)

        if truths is not None:
            pp.axhline(truths[k], color='k')

def plot_kombine_chains(chain, truths=None, mean=True, fburnin=0):
    """Like :func:`plot_emcee_chains` but for kombine.
    
    """
    plot_emcee_chains(np.transpose(chain, (1,0,2)), truths=truths, mean=mean, fburnin=fburnin)

def plot_emcee_chains_one_fig(chain, fburnin=None):
    """Plots a single-figure representation of the chain evolution of the
    given chain.  The figure shows the evolution of the mean of each
    coordinate of the ensemble, normalised to zero-mean, unit-standard
    deviation.

    :param chain: The sampler chain, of shape ``(nwalkers, niter,
      nparams)`` 

    :param fburnin: If not ``None``, refers to the fraction of samples
      to discard at the beginning of the chain.

    """

    nk = chain.shape[2]
    if fburnin is None:
        istart = 0
    else:
        istart = int(round(fburnin*chain.shape[1]))

    for k in range(nk):
        mus = np.mean(chain[:,istart:,k], axis=0)
        mu = np.mean(mus)
        sigma = np.std(mus)

        pp.plot((mus - mu)/sigma)

def plot_kombine_chains_one_fig(chain, fburnin=None):
    """Like :func:`plot_emcee_chains_one_fig` but for kombine.

    """
    plot_emcee_chains_one_fig(np.transpose(chain, (1,0,2)), fburnin=fburnin)

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

    The algorithm uses a two-step process (see `this document
    <https://dcc.ligo.org/LIGO-P1400054/public>`_) so that the
    resulting credible areas will be unbiased.

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

    Npts=pts.shape[0]

    kde_pts = pts[:Npts/2,:]
    den_pts = pts[Npts/2:,:]

    Nkde = kde_pts.shape[0]
    Nden = den_pts.shape[0]

    kde=ss.gaussian_kde(kde_pts.T)
    den=kde(den_pts.T)
    densort=np.sort(den)[::-1]

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
        ilevel = int(Nden*level + 0.5)
        if ilevel >= Nden:
            ilevel = Nden-1
        zvalues.append(densort[ilevel])

    pp.contour(XS, YS, ZS, zvalues, colors=colors, cmap=cmap, *args, **kwargs)

def greedy_kde_areas_2d(pts, levels, Nx=100, Ny=100, truth=None):
    """Returns an estimate of the area within the given credible levels
    for the posterior represented by ``pts``.  

    The algorithm uses a two-step process (see `this document
    <https://dcc.ligo.org/LIGO-P1400054/public>`_) so that the
    resulting credible areas will be unbiased.

    :param pts: An ``(Npts, 2)`` array giving samples from the
      posterior.  The algorithm assumes that each point is an
      independent draw from the posterior.

    :param levels: The credible levels for which the areas are
      desired.

    :param Nx: The number of subdivisions along the first parameter to
      be used for the credible area integral.

    :param Ny: The number of subdivisions along the second parameter
      to be used for the credible area integral.

    :param truth: If given, then the area contained within the
      posterior contours that are more probable than the posterior
      evaluated at ``truth`` will be returned.  Also, the credible
      level that corresponds to truth is returned.  The area quantity
      is sometimes called the 'searched area', since it is the area a
      greedy search algorithm will cover before finding the true
      values.

    :return: If ``truth`` is None, ``areas``, an array of the same
      shape as ``levels`` giving the credible areas; if ``truth`` is
      not ``None`` then ``(areas, searched_area, p_value)``.

    """

    pts = np.random.permutation(pts)

    mu = np.mean(pts, axis=0)
    cov = np.cov(pts, rowvar=0)

    L = np.linalg.cholesky(cov)
    detL = L[0,0]*L[1,1]

    pts = np.linalg.solve(L, (pts - mu).T).T

    if truth is not None:
        truth = np.linalg.solve(L, truth-mu)

    Npts = pts.shape[0]
    kde_pts = pts[:Npts/2, :]
    den_pts = pts[Npts/2:, :]

    kde = ss.gaussian_kde(kde_pts.T)
    den = kde(den_pts.T)
    densort = np.sort(den)[::-1]

    xs = np.linspace(np.min(pts[:,0]), np.max(pts[:,0]), Nx)
    ys = np.linspace(np.min(pts[:,1]), np.max(pts[:,1]), Ny)

    dx = xs[1]-xs[0]
    dy = ys[1]-ys[0]

    xmids = 0.5*(xs[:-1] + xs[1:])
    ymids = 0.5*(ys[:-1] + ys[1:])

    XMIDS, YMIDS = np.meshgrid(xmids, ymids)
    ZS = kde(np.row_stack((XMIDS.flatten(), YMIDS.flatten()))).reshape(XMIDS.shape)

    areas = []
    for l in levels:
        d = densort[int(round(l*den_pts.shape[0]))]
        count = np.sum(ZS > d)
        areas.append(dx*dy*count*detL)

    if truth is not None:
        td = kde(truth)
        count = np.sum(ZS > td)
        tarea = dx*dy*count*detL
        index = bisect.bisect(densort[::-1], td)
        p_value = 1-float(index)/densort.shape[0]
        return areas, tarea, p_value
    else:
        return areas
    

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
      domain of the PDF.

    :param log: If ``True``, plot a PDF for ``log(pts)`` instead of
      ``pts``.

    :return: ``(xs, ys)``, the coordinates of the plotted line.

    """

    sigma = np.std(pts)
    xs = None
    ys = None

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
        ys = kde(xs)+kde(xs+period)+kde(xs-period)
        pp.plot(xs, ys, *args, **kwargs)
        return xs, ys
    elif low is not None or high is not None:
        kde=bk.Bounded_kde(pts, low=low, high=high)
        ys = kde(xs)
        pp.plot(xs, ys, *args, **kwargs)
        return xs, ys
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

        kde=ss.gaussian_kde(lpts)

        xs = np.exp(np.linspace(xmin, xmax, N))
        ys = kde(np.log(xs))
        pp.plot(xs, ys, *args, **kwargs)
        pp.xscale('log')
        return xs, ys
    else:
        kde=ss.gaussian_kde(pts)
        ys = kde(xs)
        pp.plot(xs, ys, *args, **kwargs)
        return xs, ys

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

def plot_kde_posterior_2d(pts, xmin=None, xmax=None, ymin=None, ymax=None, Nx=100, Ny=100, cmap=None, log=False, logspace=False):
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


    
