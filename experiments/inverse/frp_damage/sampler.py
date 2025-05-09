import numpy as np

from probability import Likelihood
from probability.multivariate import Gaussian


class EllipticalSliceSampler:
    def __init__(
        self,
        *,
        prior,
        likelihood,
        n_sample,
        n_burn,
        start_value=None,
        seed=None,
        tempering=None,
        recompute_logpdf=False,
        return_info=False
    ):
        assert isinstance(prior, Gaussian)
        assert isinstance(likelihood, Likelihood)
        self.prior = prior
        self.likelihood = likelihood

        self.n_sample = n_sample
        self.n_burn = n_burn
        if start_value is None:
            self.start_value = np.zeros(len(self.prior))
        else:
            self.start_value = start_value
        self._rng = np.random.default_rng(seed)
        self.scaling = 1.0
        self.tempering = tempering
        self.recompute_logpdf = recompute_logpdf
        self.return_info = return_info

    def __call__(self):
        f = self.start_value

        if self.tempering is None:
            temp = 1.0
        else:
            temp = self.tempering(0)
            assert temp == 0.0

        logpdf = temp * self.likelihood.calc_logpdf(f)
        samples = np.zeros((self.n_sample + 1, len(self.prior)))
        samples[0] = f

        if self.return_info:
            logpdfs = np.zeros((self.n_sample + 1))
            logpdfs[0] = logpdf
            temperatures = np.zeros((self.n_sample + 1))
            temperatures[0] = temp

        for i in range(1, self.n_sample + 1):
            nu = self.prior.calc_sample(self._rng)
            u = self._rng.uniform()

            logy = logpdf + np.log(u)

            theta = self._rng.uniform(0, 2 * np.pi)
            theta_min = theta - 2 * np.pi
            theta_max = theta

            if self.tempering is not None:
                old_temp = temp
                temp = self.tempering(i)
                recompute_logpdf = self.recompute_logpdf or old_temp != temp
            else:
                recompute_logpdf = self.recompute_logpdf

            if recompute_logpdf:
                logpdf = self.likelihood.calc_logpdf(f)

            for j in range(100):
                f_prop = f * np.cos(theta) + nu * np.sin(theta)
                logpdf_prop = temp * self.likelihood.calc_logpdf(f_prop)

                if logpdf_prop > logy:
                    break

                if theta < 0:
                    theta_min = theta
                else:
                    theta_max = theta
                theta = self._rng.uniform(theta_min, theta_max)

            f = f_prop
            logpdf = logpdf_prop
            samples[i] = f

            if self.return_info:
                logpdfs[i] = logpdf
                temperatures[i] = temp

        if self.return_info:
            info = {"loglikelihood": logpdfs, "temperature": temperatures}
            return samples, info
        else:
            return samples
