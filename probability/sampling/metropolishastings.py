import os
import numpy as np
from scipy.sparse import issparse
from scipy.special import logsumexp
import pickle

from probability import Distribution, IndependentJoint, RejectConditional
from probability.multivariate import Gaussian as MVGaussian
from probability.univariate import (
    LogGaussian,
    Uniform,
    Gaussian as UVGaussian,
)

__all__ = [
    "MetropolisHastingsSampler",
    "RandomWalkMetropolisSampler",
    "IndependenceSampler",
]


class MetropolisHastingsSampler:
    def __init__(
        self,
        *,
        target,
        proposal_rw=None,
        proposal_ind=None,
        beta=None,
        n_sample,
        n_burn,
        start_value=None,
        seed=None,
        tune=True,
        tune_interval=100,
        tempering=None,
        recompute_logpdf=False,
        return_info=False,
        checkpoint=None,
        checkpoint_interval=1000,
    ):
        assert isinstance(target, Distribution)

        # assumed structure: (1 - beta) * q_rw + beta * q_ind
        if proposal_ind is None:
            # only random walk metropolis
            assert isinstance(proposal_rw, Distribution)
            assert beta is None or beta == 0.0
            self.beta = 0.0
        elif proposal_rw is None:
            # only independence sampler
            assert isinstance(proposal_ind, Distribution)
            assert beta is None or beta == 1.0
            self.beta = 1.0
        else:
            # mix of the two
            assert isinstance(proposal_rw, Distribution)
            assert isinstance(proposal_ind, Distribution)
            assert 0.0 < beta < 1.0
            self.beta = beta

        assert 0.0 <= self.beta <= 1.0

        self.target = target
        self.proposal_rw = proposal_rw
        self.proposal_ind = proposal_ind

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
        self.checkpoint = checkpoint
        self.checkpoint_interval = checkpoint_interval

    def __call__(self):

        if self.checkpoint is None:
            start = 0
        else:
            start = self._load_checkpoint()

        if start == 0:
            xi = self.start_value
        else:
            xi = self.samples[start]

        if self.tempering is None:
            temp = 1.0
        else:
            temp = self.tempering(start)
            self.target.set_temperature(temp)

        logpdf = self.target.calc_logpdf(xi)

        if start == 0:
            self.samples = np.zeros((self.n_sample + 1, len(self.target)))
            self.samples[0] = xi

            if self.return_info:
                self.logpdfs = np.zeros((self.n_sample + 1))
                self.logpdfs[0] = logpdf
                self.temperatures = np.zeros((self.n_sample + 1))
                self.temperatures[0] = temp
            else:
                self.logpdfs = None
                self.temperatures = None

        accept_rate = 0.0

        for i in range(start + 1, self.n_sample + 1):
            if self.beta == 0.0:
                self.proposal_rw.update_mean(xi)
                xi_prop = self.proposal_rw.calc_sample(self._rng)
            elif self.beta == 1.0:
                xi_prop = self.proposal_ind.calc_sample(self._rng)
            else:
                if self._rng.uniform() < self.beta:
                    xi_prop = self.proposal_ind.calc_sample(self._rng)
                else:
                    self.proposal_rw.update_mean(xi)
                    xi_prop = self.proposal_rw.calc_sample(self._rng)

            if self.tempering is not None:
                old_temp = temp
                temp = self.tempering(i)
                self.target.set_temperature(temp)
                recompute_logpdf = self.recompute_logpdf or old_temp != temp
            else:
                recompute_logpdf = self.recompute_logpdf

            if recompute_logpdf:
                logpdf = self.target.calc_logpdf(xi)

            try:
                logpdf_prop = self.target.calc_logpdf(xi_prop)
            except Exception as error:
                if i < self.n_burn:
                    print("Exception caught:", error)
                    print("Still in burn-in phase, continuing MCMC run")
                    print("Setting logpdf_prop = -inf")
                    logpdf_prop = -np.inf
                else:
                    raise error

            if self.beta == 0.0:
                logalpha = logpdf_prop - logpdf
            elif self.beta == 1.0:
                logq_ind = self.proposal_ind.calc_logpdf(xi)
                logq_ind_prop = self.proposal_ind.calc_logpdf(xi_prop)
                logalpha = logpdf_prop - logpdf - logq_ind_prop + logq_ind
            else:
                logq_ind = self.proposal_ind.calc_logpdf(xi)
                logq_ind_prop = self.proposal_ind.calc_logpdf(xi_prop)

                self.proposal_rw.update_mean(xi_prop)
                logq_rw = self.proposal_rw.calc_logpdf(xi)
                self.proposal_rw.update_mean(xi)
                logq_rw_prop = self.proposal_rw.calc_logpdf(xi_prop)

                logq_tot = logsumexp(
                    [
                        np.log(1 - self.beta) + logq_rw,
                        np.log(self.beta) + logq_ind,
                    ]
                )
                logq_tot_prop = logsumexp(
                    [
                        np.log(1 - self.beta) + logq_rw_prop,
                        np.log(self.beta) + logq_ind_prop,
                    ]
                )
                logalpha = logpdf_prop - logpdf - logq_tot_prop + logq_tot

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

            self.samples[i] = xi

            if self.return_info:
                self.logpdfs[i] = logpdf
                self.temperatures[i] = temp

            if i % self.tune_interval == 0:
                print("MCMC sample {} of {}".format(i, self.n_sample))
                print(xi)
                print(logpdf, temp)
                print("Accept rate:", accept_rate)
                print("")

                if self.tune and i <= self.n_burn:
                    if self.beta == 0.0 and isinstance(self.proposal_rw, MVGaussian):
                        if accept_rate > 0.1:
                            sample_batch = self.samples[i - self.tune_interval : i]
                            shaping = self._recompute_shaping(sample_batch)
                            self._shape_proposal(self.proposal_rw, shaping)

                    if self.beta != 1.0:
                        oldscaling = self.scaling
                        newscaling = self._recompute_scaling(oldscaling, accept_rate)

                        if not np.isclose(oldscaling, newscaling):
                            factor = newscaling / oldscaling
                            self._scale_proposal(self.proposal_rw, factor)
                            self.scaling = newscaling

                accept_rate = 0.0

            if i % self.checkpoint_interval == 0:
                self._save_checkpoint(i)

        self._remove_checkpoint()

        if self.return_info:
            info = {
                "loglikelihood": self.logpdfs,
                "temperature": self.temperatures,
            }
            return self.samples, info
        else:
            return self.samples

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

    def _recompute_shaping(self, samples):
        sample_cov = np.cov(samples.T)
        prop_cov = self.proposal_rw.calc_cov()

        if issparse(prop_cov):
            prop_cov = prop_cov.toarray()

        l_sample, Q_sample = np.linalg.eigh(sample_cov)
        l_prop, Q_prop = np.linalg.eigh(prop_cov)

        log_det_sample = np.sum(np.log(l_sample))
        log_det_prop = np.sum(np.log(l_prop))
        scale = np.exp((log_det_sample - log_det_prop) / (2 * len(l_prop)))
        log_l_ratio = 0.5 * (np.log(l_sample) - np.log(l_prop))

        shaping = Q_sample @ np.diag(np.exp(log_l_ratio)) @ Q_prop.T / scale
        return shaping

    def _scale_proposal(self, proposal, factor):
        if isinstance(proposal, IndependentJoint):
            for dist in proposal.distributions:
                self._scale_proposal(dist, factor)
        elif isinstance(proposal, RejectConditional):
            self._scale_proposal(proposal.latent, factor)
        elif isinstance(proposal, UVGaussian):
            std = proposal.calc_std()
            proposal.update_std(np.sqrt(factor) * std)
        elif isinstance(proposal, MVGaussian):
            cov = proposal.calc_cov()
            proposal.update_cov(factor * cov)
        elif isinstance(proposal, LogGaussian):
            logstd = proposal.calc_latent_std()
            proposal.update_latent_std(np.sqrt(factor) * logstd)
        elif isinstance(proposal, Uniform):
            width = proposal.calc_width()
            proposal.update_width(np.sqrt(factor) * width)
        else:
            raise ValueError

    def _shape_proposal(self, proposal, factor):
        if isinstance(proposal, MVGaussian):
            cov = proposal.calc_cov()
            noise = 1e-8 * np.identity(cov.shape[0])

            try:
                proposal.update_cov(factor @ cov @ factor.T + noise)
            except Exception as error:
                print("Exception caught:", error)
                print("Not reshaping covariance")
        else:
            raise ValueError

    def _save_checkpoint(self, i):
        if self.checkpoint is None:
            return

        state = {
            "i": i,
            "samples": self.samples,
            "logpdfs": self.logpdfs,
            "temperatures": self.temperatures,
            "proposal_rw": self.proposal_rw,
            "proposal_ind": self.proposal_ind,
            "scaling": self.scaling,
            "rng": self._rng,
        }

        os.makedirs(os.path.dirname(self.checkpoint), exist_ok=True)

        with open(self.checkpoint, "wb") as f:
            pickle.dump(state, f)

        rng_state = hex(self._rng.bit_generator.state["state"]["state"])
        print("Saved checkpoint with", i, "samples and rng:", rng_state)
        print("")

    def _load_checkpoint(self):
        if self.checkpoint is None:
            return 0
        elif not os.path.isfile(self.checkpoint):
            return 0

        with open(self.checkpoint, "rb") as f:
            state = pickle.load(f)

        i = state["i"]
        self.samples = state["samples"]
        self.logpdfs = state["logpdfs"]
        self.temperatures = state["temperatures"]
        self.proposal_rw = state["proposal_rw"]
        self.proposal_ind = state["proposal_ind"]
        self.scaling = state["scaling"]
        self._rng = state["rng"]

        rng_state = hex(self._rng.bit_generator.state["state"]["state"])
        print("Loaded checkpoint with", i, "samples and rng:", rng_state)
        print("")

        return i

    def _remove_checkpoint(self):
        if self.checkpoint is None:
            return

        if os.path.isfile(self.checkpoint):
            os.remove(self.checkpoint)
            print("Removed checkpoint")
            print("")


class RandomWalkMetropolisSampler(MetropolisHastingsSampler):
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
        return_info=False,
        checkpoint=None,
        checkpoint_interval=1000,
    ):

        super().__init__(
            target=target,
            proposal_rw=proposal,
            proposal_ind=None,
            beta=0.0,
            n_sample=n_sample,
            n_burn=n_burn,
            start_value=start_value,
            seed=seed,
            tune=tune,
            tune_interval=tune_interval,
            tempering=tempering,
            recompute_logpdf=recompute_logpdf,
            return_info=return_info,
            checkpoint=checkpoint,
            checkpoint_interval=checkpoint_interval,
        )


class IndependenceSampler(MetropolisHastingsSampler):
    def __init__(
        self,
        *,
        target,
        proposal,
        n_sample,
        n_burn,
        start_value=None,
        seed=None,
        recompute_logpdf=False,
        return_info=False,
        checkpoint=None,
        checkpoint_interval=1000,
    ):

        super().__init__(
            target=target,
            proposal_rw=None,
            proposal_ind=proposal,
            beta=1.0,
            n_sample=n_sample,
            n_burn=n_burn,
            start_value=start_value,
            seed=seed,
            tune=False,
            recompute_logpdf=recompute_logpdf,
            return_info=return_info,
            checkpoint=checkpoint,
            checkpoint_interval=checkpoint_interval,
        )
