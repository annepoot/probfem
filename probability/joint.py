import numpy as np

from probability.distribution import (
    Distribution,
    UnivariateDistribution,
    MultivariateDistribution,
)
from probability.multivariate import IsotropicGaussian


__all__ = ["IndependentJoint"]


class IndependentJoint(MultivariateDistribution):
    def __init__(self, *distributions):
        for dist in distributions:
            assert isinstance(dist, Distribution)

        self.distributions = distributions

    def __len__(self):
        length = 0
        for dist in self.distributions:
            length += len(dist)
        return length

    def update_mean(self, mean):
        offset = 0
        for dist in self.distributions:
            l = len(dist)
            if isinstance(dist, UnivariateDistribution):
                assert l == 1
                dist.update_mean(mean[offset])
            else:
                dist.update_mean(mean[offset : offset + l])
            offset += l
        assert offset == len(self)

    def update_cov(self, cov):
        offset = 0
        for dist in self.distributions:
            l = len(dist)
            if isinstance(dist, UnivariateDistribution):
                assert l == 1
                dist.update_std(np.sqrt(cov[offset, offset]))
            elif isinstance(dist, IsotropicGaussian):
                std = np.sqrt(cov[offset, offset])
                assert np.allclose(
                    cov[offset : offset + l, offset : offset + l],
                    np.diag(std**2 * np.ones(l)),
                )
                dist.update_std(std)
            else:
                dist.update_cov(cov[offset : offset + l, offset : offset + l])
            offset += l
        assert offset == len(self)

    def calc_sample(self, seed):
        length = len(self)
        sample = np.zeros(length)
        offset = 0
        for dist in self.distributions:
            l = len(dist)
            sample[offset : offset + l] = dist.calc_sample(seed)
            offset += l
        assert offset == length
        return sample

    def calc_samples(self, n, seed):
        length = len(self)
        samples = np.zeros((n, length))
        offset = 0
        for dist in self.distributions:
            l = len(dist)
            samples[:, offset : offset + l] = dist.calc_samples(n, seed)
            offset += l
        assert offset == length
        return samples

    def calc_pdf(self, x):
        pdf = 1
        offset = 0
        for dist in self.distributions:
            l = len(dist)
            if isinstance(dist, UnivariateDistribution):
                assert l == 1
                pdf *= dist.calc_pdf(x[offset])
            else:
                pdf *= dist.calc_pdf(x[offset : offset + l])
            offset += l
        assert offset == len(self)
        return pdf

    def calc_logpdf(self, x):
        logpdf = 0
        offset = 0
        for dist in self.distributions:
            l = len(dist)
            if isinstance(dist, UnivariateDistribution):
                assert l == 1
                logpdf += dist.calc_logpdf(x[offset])
            else:
                logpdf += dist.calc_logpdf(x[offset : offset + l])
            offset += l
        assert offset == len(self)
        return logpdf

    def calc_mean(self):
        length = len(self)
        mean = np.zeros(length)
        offset = 0
        for dist in self.distributions:
            l = len(dist)
            mean[offset : offset + l] = dist.calc_mean()
            offset += l
        assert offset == length
        return mean

    def calc_cov(self):
        length = len(self)
        cov = np.zeros((length, length))
        offset = 0
        for dist in self.distributions:
            l = len(dist)
            if isinstance(dist, UnivariateDistribution):
                assert l == 1
                cov[offset, offset] = dist.calc_std() ** 2
            else:
                cov[offset : offset + l, offset : offset + l] = dist.calc_cov()
            offset += l
        assert offset == length
        return cov

    def calc_std(self):
        length = len(self)
        std = np.zeros(length)
        offset = 0
        for dist in self.distributions:
            l = len(dist)
            std[offset : offset + l] = dist.calc_std()
            offset += l
        assert offset == length
        return std
