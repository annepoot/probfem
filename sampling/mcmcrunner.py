import numpy as np
from myjive.util.proputils import check_dict, split_off_type

from probability.distribution import Distribution
from probability.multivariate import IsotropicGaussian, DiagonalGaussian

__all__ = ["MCMCRunner"]


class MCMCRunner:
    def __init__(
        self,
        *,
        target,
        proposal,
        nsample,
        startValue=None,
        seed=None,
        tune=True,
        tuneInterval=100,
    ):
        # Validate input arguments
        check_dict(self, target, ["type"])
        check_dict(self, proposal, ["type"])

        target_cls, target_kws = split_off_type(target)
        proposal_cls, proposal_kws = split_off_type(proposal)

        assert issubclass(target_cls, Distribution)
        assert issubclass(proposal_cls, Distribution)

        self._target = target_cls(**target_kws)
        self._proposal = proposal_cls(**proposal_kws)

        self._nvar = len(self._target)
        self._nsample = nsample
        self._start = np.zeros(self._nvar) if startValue is None else startValue
        self._rng = np.random.default_rng(seed)
        self._tune = tune
        self._tune_interval = tuneInterval
        self._scaling = 1.0

    def __call__(self):
        xi = self._start

        logpdf = self._target.calc_logpdf(xi)
        samples = np.zeros((self._nsample + 1, self._nvar))
        samples[0] = xi

        accept_rate = 0.0

        for i in range(1, self._nsample + 1):
            if self._tune and i % self._tune_interval == 0:
                print("MCMC sample {} of {}".format(i, self._nsample))
                oldscaling = self._scaling
                newscaling = self._recompute_scaling(oldscaling, accept_rate)

                if not np.isclose(oldscaling, newscaling):
                    factor = newscaling / oldscaling
                    if isinstance(self._proposal, IsotropicGaussian):
                        std = self._proposal.calc_std()
                        self._proposal.update_std(np.sqrt(factor) * std)
                    elif isinstance(self._proposal, DiagonalGaussian):
                        diag = self._proposal.calc_diag()
                        self._proposal.update_diag(factor * diag)
                    else:
                        cov = self._proposal.calc_cov()
                        self._proposal.update_cov(factor * cov)
                    self._scaling = newscaling
                accept_rate = 0.0

            self._proposal.update_mean(xi)
            xi_prop = self._proposal.calc_sample(self._rng)
            logpdf_prop = self._target.calc_logpdf(xi_prop)
            logalpha = logpdf_prop - logpdf

            if logalpha < 0:
                if self._rng.uniform() < np.exp(logalpha):
                    accept = True
                else:
                    accept = False
            else:
                accept = True

            if accept:
                xi = xi_prop
                logpdf = logpdf_prop
                accept_rate += 1 / self._tune_interval

            samples[i] = xi

        return samples

    def _recompute_scaling(self, scaling, accept_rate):
        print("Accept rate:", accept_rate)
        print("Old scaling:", scaling)
        if accept_rate < 0.001:
            scaling *= 0.1
        elif accept_rate < 0.05:
            scaling *= 0.5
        elif accept_rate < 0.2:
            scaling *= 0.9

        if accept_rate > 0.95:
            scaling *= 10
        elif accept_rate > 0.75:
            scaling *= 2
        elif accept_rate > 0.4:
            scaling *= 1.2
        print("New scaling:", scaling)
        print("")
        return scaling
