import numpy as np

from myjive.util.proputils import check_dict

from probability.distribution import Distribution
from probability.multivariate import IsotropicGaussian, DiagonalGaussian


__all__ = ["MCMCRunner"]


class MCMCRunner:
    def __init__(
        self,
        *,
        target,
        proposal,
        n_sample,
        start_value=None,
        seed=None,
        tune=True,
        tune_interval=100,
        recompute_logpdf=False
    ):
        assert isinstance(target, Distribution)
        assert isinstance(proposal, Distribution)
        self.target = target
        self.proposal = proposal

        self.n_sample = n_sample
        if start_value is None:
            self.start_value = np.zeros(len(self.target))
        else:
            self.start_value = start_value
        self._rng = np.random.default_rng(seed)
        self.tune = tune
        self.tune_interval = tune_interval
        self.scaling = 1.0
        self.recompute_logpdf = recompute_logpdf

    def __call__(self):
        xi = self.start_value

        logpdf = self.target.calc_logpdf(xi)
        samples = np.zeros((self.n_sample + 1, len(self.target)))
        samples[0] = xi

        accept_rate = 0.0

        for i in range(1, self.n_sample + 1):
            if i % self.tune_interval == 0:
                print("MCMC sample {} of {}".format(i, self.n_sample))
                print(xi)
                print("Accept rate:", accept_rate)

                if self.tune:
                    oldscaling = self.scaling
                    newscaling = self._recompute_scaling(oldscaling, accept_rate)

                    if not np.isclose(oldscaling, newscaling):
                        factor = newscaling / oldscaling
                        if isinstance(self.proposal, IsotropicGaussian):
                            std = self.proposal.calc_std()
                            assert np.allclose(std, std[0])
                            self.proposal.update_std(np.sqrt(factor) * std[0])
                        elif isinstance(self.proposal, DiagonalGaussian):
                            diag = self.proposal.calc_diag()
                            self.proposal.update_diag(factor * diag)
                        else:
                            cov = self.proposal.calc_cov()
                            self.proposal.update_cov(factor * cov)
                        self.scaling = newscaling

                    accept_rate = 0.0

            self.proposal.update_mean(xi)
            xi_prop = self.proposal.calc_sample(self._rng)
            logpdf_prop = self.target.calc_logpdf(xi_prop)

            if self.recompute_logpdf:
                logpdf = self.target.calc_logpdf(xi)

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
                accept_rate += 1 / self.tune_interval

            samples[i] = xi

        return samples

    def _recompute_scaling(self, scaling, accept_rate):
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
