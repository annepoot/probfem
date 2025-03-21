import numpy as np

from probability.distribution import MultivariateDistribution


__all__ = ["RejectConditional"]


class RejectConditional(MultivariateDistribution):
    def __init__(self, *, latent, reject_if):
        self.latent = latent
        self.reject_if = reject_if

    def __len__(self):
        return len(self.latent)

    def update_mean(self, mean):
        self.latent.update_mean(mean)

    def update_cov(self, cov):
        self.latent.update_cov(cov)

    def calc_sample(self, seed):
        for _ in range(10000):
            sample = self.latent.calc_sample(seed)
            reject = self.reject_if(sample)
            if not reject:
                return sample
        assert False

    def calc_samples(self, n, seed):
        samples = self.latent.calc_samples(n, seed)
        for i, sample in enumerate(samples):
            reject = self.reject_if(sample)
            if reject:
                samples[i] = self.calc_sample(seed)
        return samples

    def calc_pdf(self, x):
        reject = self.reject_if(x)
        if reject:
            return 0.0
        else:
            return self.latent.calc_pdf(x)

    def calc_logpdf(self, x):
        reject = self.reject_if(x)
        if reject:
            return -np.inf
        else:
            return self.latent.calc_logpdf(x)

    def calc_mean(self):
        raise NotImplementedError

    def calc_cov(self):
        raise NotImplementedError

    def calc_std(self):
        raise NotImplementedError
