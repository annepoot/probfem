import numpy as np

from probability.multivariate import Gaussian
from ..distribution import Distribution
from .mean_functions import MeanFunction, ZeroMeanFunction, MeanFunctionSum
from .covariance_functions import CovarianceFunction, CovarianceFunctionSum

__all__ = ["GaussianProcess"]


class GaussianProcess(Distribution):
    def __init__(self, mean, cov):
        self.update_mean(mean)
        self.update_cov(cov)

    def __add__(self, gp2):
        if not isinstance(gp2, GaussianProcess):
            raise TypeError

        mean = MeanFunctionSum(self.mean, gp2.mean)
        cov = CovarianceFunctionSum(self.cov, gp2.cov)

        return GaussianProcess(mean, cov)

    def __len__(self):
        return self.len

    def update_mean(self, mean):
        if mean is None:
            mean = ZeroMeanFunction()

        if not isinstance(mean, MeanFunction):
            raise TypeError
        self.mean = mean

    def update_cov(self, cov):
        if not isinstance(cov, CovarianceFunction):
            raise TypeError
        self.cov = cov

    def calc_mean(self, x):
        return self.mean.calc_mean(x)

    def calc_cov(self, x1, x2):
        return self.cov.calc_cov(x1, x2)

    def calc_std(self, x):
        cov = self.calc_cov(x, x)
        return np.sqrt(np.diagonal(cov))

    def calc_sample(self, x, seed):
        mean = self.calc_mean(x)
        cov = self.calc_cov(x, x)
        sqrtcov = np.linalg.cholesky(cov)
        rng = np.random.default_rng(seed)
        return mean + sqrtcov @ rng.standard_normal(len(x))

    def calc_samples(self, x, n, seed):
        mean = self.calc_mean(x)
        cov = self.calc_cov(x, x)
        sqrtcov = np.linalg.cholesky(cov)
        rng = np.random.default_rng(seed)
        return np.tile(mean, (n, 1)).T + sqrtcov @ rng.standard_normal((len(x), n))

    def calc_pdf(self, x):
        return np.exp(self.evaluate_logpdf(x))

    def calc_logpdf(self, x, value):
        # TODO: improve efficiency via backsubstitution
        mean = self.calc_mean(x)
        cov = self.calc_cov(x, x)
        sqrtcov = np.linalg.cholesky(cov)
        logdet = 2 * np.sum(np.log(np.diagonal(sqrtcov)))
        potential = 1 / 2 * (value - mean) @ np.linalg.solve(cov, (value - mean))
        return -len(x) / 2 * np.log(2 * np.pi) - len(x) / 2 * logdet - potential

    def evaluate_at(self, x):
        mean = self.calc_mean(x)
        cov = self.calc_cov(x, x)
        return Gaussian(mean, cov, allow_singular=True)
