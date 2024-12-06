import numpy as np

from ..distribution import UnivariateDistribution
from .gaussian import Gaussian

__all__ = ["LogGaussian"]


class LogGaussian(UnivariateDistribution):
    def __init__(self, logmean, logstd, allow_logscale_access=False):
        self.latent = Gaussian(logmean, logstd)
        self.allow_logscale_access = allow_logscale_access

    def update_latent_mean(self, logmean):
        self.latent.update_mean(logmean)

    def update_mean(self, mean):
        # This is a bit hacky, because log(mean) is not the true mean
        if self.allow_logscale_access:
            logmean = np.log(mean)
            self.update_latent_mean(logmean)
        else:
            raise NotImplementedError

    def update_latent_std(self, logstd):
        self.latent.update_std(logstd)

    def update_std(self, std):
        raise NotImplementedError

    def calc_latent_mean(self):
        return self.latent.calc_mean()

    def calc_mean(self):
        # This is a bit hacky, because log(mean) is not the true mean
        if self.allow_logscale_access:
            return np.exp(self.calc_latent_mean())
        else:
            raise NotImplementedError

    def calc_true_mean(self):
        logmean = self.latent.calc_mean()
        logvar = self.latent.calc_var()
        return np.exp(logmean + 0.5 * logvar)

    def calc_latent_var(self):
        return self.latent.calc_var()

    def calc_true_var(self):
        logmean = self.latent.calc_mean()
        logvar = self.latent.calc_var()
        return (np.exp(logvar) - 1) * np.exp(2 * logmean + logvar)

    def calc_latent_std(self):
        return self.latent.calc_std()

    def calc_std(self):
        raise NotImplementedError

    def calc_true_std(self):
        return np.sqrt(self.calc_true_var())

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
