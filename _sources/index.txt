.. plotutils documentation master file, created by
   sphinx-quickstart on Sat Feb  9 18:16:17 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.



Welcome to plotutils's documentation!
=====================================

``plotutils`` contains utilities that are designed to make working
with samples from 1- and 2-D probability density functions easier with
``matplotlib``.  In particular, from a set of samples drawn from some
posterior distribution you can

* Plot 1- and 2-D histogram estimates of PDFs using
  :func:`plotutils.plotutils.plot_histogram_posterior` and
  :func:`plotutils.plotutils.plot_histogram_posterior_2d`.  The bin sizes will be set
  automatically using a recipe that attempts to minimize the expected
  squared error between the histogram PDF and the true PDF.
* Plot 1- and 2-D kernel density estimates of PDFs using
  :func:`plotutils.plotutils.plot_kde_posterior` and
  :func:`plotutils.plotutils.plot_kde_posterior_2d`.
* Plot 1- and 2-D probability intervals containing a specified
  fraction of the total posterior.  In 2-D, the interval will be
  accumulated greedily---starting with the most likely sample points,
  and working down in posterior probability---using either the
  histogram or kernel density estimate of the PDF.  See
  :func:`plotutils.plotutils.plot_interval`,
  :func:`plotutils.plotutils.plot_greedy_histogram_interval_2d`, and
  :func:`plotutils.plotutils.plot_greedy_kde_interval_2d`.
* Estimate autocorrelation functions and autocorrelation lengths of
  one-dimensional series.  See
  :func:`plotutils.autocorr.autocorrelation_function` and
  :func:`plotutils.autocorr.autocorrelation_length_estimate`.

To install ``plotutils``, simply execute ``python setup.py install``
in the root directory.

You can find the source for the ``plotutils`` package at
http://github.com/farr/plotutils.

Contents:

.. toctree::
   :maxdepth: 2

   plotutils


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

