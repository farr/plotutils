import autocorr as ac
import bz2
import emcee
import numpy as np
import os
import os.path as op
import pickle

def load_runner(dir, fname='runner.pkl.bz2'):
    """Loads the saved runner from the given directory.

    """
    with bz2.BZ2File(op.join(dir, fname), 'r') as inp:
        runner = pickle.load(inp)
    return runner

class EnsembleSamplerRunner(object):
    """Runner object for an emcee sampler.

    """

    def __init__(self, sampler, pts):
        """Initialise the runner with the given sampler and initial ensemble
        position.

        """

        self.sampler = sampler
        self.result = pts
        self._first_step = True

        self.thin = 1

    @property
    def chain(self):
        """The current state of the sampler's chain.

        """
        return self.sampler.chain

    @property
    def lnprobability(self):
        """The current state of the sampler's lnprobability.

        """
        return self.sampler.lnprobability

    @property
    def thin_chain(self):
        """Return a thinned chain (if possible), using
        :func:`ac.emcee_thinned_chain`

        """
        return ac.emcee_thinned_chain(self.chain)

    @property
    def thin_flatchain(self):
        """Returns a thinned chain that has been flattened.

        """
        tc = self.thin_chain
        return tc.reshape((-1, tc.shape[2]))

    @property
    def acls(self):
        """Return the estimate of the current chain's autocorrelation lengths,
        using :func:`plotutils.autocorr.emcee_chain_autocorrelation_lengths`.

        """
        return ac.emcee_chain_autocorrelation_lengths(self.chain)

    def save_state(self, dir, fname='runner.pkl.bz2'):
        """Stores the current state of the runner via a pickled object in the
        given directory.  The object will be pickled and bz2-compressed.

        """

        name,ext = op.split(fname)

        if not (ext == '.bz2'):
            fname += '.bz2'

        tfname = fname + '.temp'

        with bz2.BZ2File(op.join(dir, tfname), 'w') as out:
            pickle.dump(self, out)
        os.rename(op.join(dir, tfname), op.join(dir, fname))

    def run_mcmc(self, nthinsteps):
        """Run the associated sampler to produce ``nthinsteps`` worth of
        stored ensembles (i.e. the sampler will be run for
        ``nthinsteps*self.thin`` total steps).

        """

        nsteps = self.thin * nthinsteps
        
        if self._first_step:
            self.result = self.sampler.run_mcmc(self.result, nsteps, thin=self.thin)
            self._first_step = False
        else:
            self.result = self.sampler.run_mcmc(self.result[0], nsteps, lnprob0=self.result[1], thin=self.thin)

        return self.result

    def run_to_neff(self, neff, savedir=None):
        """Run the sampler, thinning as necessary, until ``neff`` effective
        ensembles are obtained.  When ``savedir`` is not ``None``, the
        sampler will be periodically saved to the given directory.

        """
        
        while self.thin_chain is None or self.thin_chain.shape[1] < neff:
            self.run_mcmc(neff)
            
            print 'Accumulated ', self.chain.shape[1], ' ensembles'
            if self.thin_chain is not None:
                print 'Equivalent to ', self.thin_chain.shape[1], ' effective ensembles'

            if savedir is not None:
                print 'Saving state...'
                self.save_state(savedir)

            if self.chain.shape[1] > 10*neff:
                self.rethin()
                print 'Thinned chain; now ', self.chain.shape[1], ' ensembles'

    def rethin(self):
        """Increase the thinning parameter by a factor of two, modifying the
        stored chain and lnprob states accordingly.  

        """

        self.sampler._chain = self.sampler._chain[:,1::2,:]
        self.sampler._lnprob = self.sampler._lnprob[:,1::2]
        self.thin *= 2
