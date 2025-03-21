import numpy as np

from probability import Distribution, IndependentJoint, RejectConditional
from probability.multivariate import IsotropicGaussian, DiagonalGaussian, Gaussian
from probability.univariate import LogGaussian, Uniform

__all__ = ["MCMCRunner"]


class MCMCRunner:
    def __init__(
        self,
        *,
        target,
        proposal,
        n_sample,
        n_burn,
        start_value=None,
        seed=None,
        tune=True,
        tune_interval=100,
        tempering=None,
        recompute_logpdf=False,
        return_info=False
    ):
        assert isinstance(target, Distribution)
        assert isinstance(proposal, Distribution)
        self.target = target
        self.proposal = proposal

        self.n_sample = n_sample
        self.n_burn = n_burn
        if start_value is None:
            self.start_value = np.zeros(len(self.target))
        else:
            self.start_value = start_value
        self._rng = np.random.default_rng(seed)
        self.tune = tune
        self.tune_interval = tune_interval
        self.scaling = 1.0
        self.tempering = tempering
        self.recompute_logpdf = recompute_logpdf
        self.return_info = return_info

    def __call__(self):
        xi = self.start_value

        if self.tempering is not None:
            temp = self.tempering(0)
            assert temp == 0.0
            self.target.set_temperature(temp)

        if self.tempering is None:
            temp = 1.0
        else:
            temp = self.tempering(0)

        logpdf = self.target.calc_logpdf(xi)
        samples = np.zeros((self.n_sample + 1, len(self.target)))
        samples[0] = xi

        if self.return_info:
            logpdfs = np.zeros((self.n_sample + 1))
            logpdfs[0] = logpdf
            temperatures = np.zeros((self.n_sample + 1))
            temperatures[0] = temp

        accept_rate = 0.0

        for i in range(1, self.n_sample + 1):
            self.proposal.update_mean(xi)
            xi_prop = self.proposal.calc_sample(self._rng)

            if self.tempering is not None:
                old_temp = temp
                temp = self.tempering(i)
                self.target.set_temperature(temp)
                recompute_logpdf = self.recompute_logpdf or old_temp != temp
            else:
                recompute_logpdf = self.recompute_logpdf

            if recompute_logpdf:
                logpdf = self.target.calc_logpdf(xi)

            logpdf_prop = self.target.calc_logpdf(xi_prop)
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

            if self.return_info:
                logpdfs[i] = logpdf
                temperatures[i] = temp

            if i % self.tune_interval == 0:
                print("MCMC sample {} of {}".format(i, self.n_sample))
                print(xi)
                print(logpdf, temp)
                print("Accept rate:", accept_rate)
                print("")

                if self.tune and i <= self.n_burn:
                    oldscaling = self.scaling
                    newscaling = self._recompute_scaling(oldscaling, accept_rate)

                    if not np.isclose(oldscaling, newscaling):
                        factor = newscaling / oldscaling
                        self._scale_proposal(self.proposal, factor)
                        self.scaling = newscaling

                accept_rate = 0.0

        if self.return_info:
            info = {"loglikelihood": logpdfs, "temperature": temperatures}
            return samples, info
        else:
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

    def _scale_proposal(self, proposal, factor):
        if isinstance(proposal, IndependentJoint):
            for dist in proposal.distributions:
                self._scale_proposal(dist, factor)
        elif isinstance(proposal, RejectConditional):
            self._scale_proposal(proposal.latent, factor)
        elif isinstance(proposal, IsotropicGaussian):
            std = proposal.calc_std()
            assert np.allclose(std, std[0])
            proposal.update_std(np.sqrt(factor) * std[0])
        elif isinstance(proposal, DiagonalGaussian):
            diag = proposal.calc_diag()
            proposal.update_diag(factor * diag)
        elif isinstance(proposal, Gaussian):
            cov = proposal.calc_cov()
            proposal.update_cov(factor * cov)
        elif isinstance(proposal, LogGaussian):
            logstd = proposal.calc_latent_std()
            proposal.update_latent_std(factor * logstd)
        elif isinstance(proposal, Uniform):
            width = proposal.calc_width()
            proposal.update_width(factor * width)
        else:
            raise ValueError
