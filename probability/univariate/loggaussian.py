import numpy as np

from ..distribution import UnivariateDistribution
from .gaussian import Gaussian

__all__ = ["LogGaussian"]


class LogGaussian(UnivariateDistribution):
    def __init__(self, logmean, logstd):
        self.latent = Gaussian(logmean, logstd)

    def update_mean(self, logmean):
        self.latent.update_mean(logmean)

    def update_std(self, logstd):
        self.latent.update_std(logstd)

    def calc_mean(self):
        logmean = self.latent.calc_mean()
        logvar = self.latent.calc_var()
        return np.exp(logmean + 0.5 * logvar)

    def calc_std(self):
        return np.sqrt(self.calc_var())

    def calc_var(self):
        logmean = self.latent.calc_mean()
        logvar = self.latent.calc_var()
        return (np.exp(logvar) - 1) * np.exp(2 * logmean + logvar)

    def calc_sample(self, seed):
        logsample = self.latent.calc_sample(seed)
        return np.exp(logsample)

    def calc_samples(self, n, seed):
        logsamples = self.latent.calc_samples(n, seed)
        return np.exp(logsamples)

    def calc_pdf(self, x):
        return self.latent.calc_pdf(np.log(x)) / x

    def calc_logpdf(self, x):
        return np.log(self.calc_pdf(x))
