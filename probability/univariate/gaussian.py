import numpy as np

from ..distribution import UnivariateDistribution
from ..multivariate import Gaussian as MVGaussian

__all__ = ["Gaussian"]


class Gaussian(UnivariateDistribution):
    def __init__(self, mean, std):
        self.update_mean(mean)
        self.update_std(std)

    def update_mean(self, mean):
        if mean is None:
            self.mean = 0.0
        elif np.isscalar(mean):
            self.mean = mean
        else:
            raise ValueError

    def update_std(self, std):
        if np.isscalar(std):
            if std < 0:
                raise ValueError
            self.std = std
        else:
            raise ValueError

    def calc_mean(self):
        return self.mean

    def calc_var(self):
        return self.std**2

    def calc_std(self):
        return self.std

    def calc_sample(self, seed):
        rng = np.random.default_rng(seed)
        return self.mean + self.std * rng.standard_normal()

    def calc_samples(self, n, seed):
        rng = np.random.default_rng(seed)
        return self.mean + self.std * rng.standard_normal(n)

    def calc_pdf(self, x):
        return np.exp(self.calc_logpdf(x))

    def calc_logpdf(self, x):
        potential = 0.5 * (x - self.mean) ** 2 / self.std**2
        return -0.5 * np.log(2 * np.pi) - np.log(self.std) - potential
