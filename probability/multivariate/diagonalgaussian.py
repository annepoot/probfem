import numpy as np
from scipy.stats import Covariance

from .gaussian import Gaussian

__all__ = ["DiagonalGaussian", "IsotropicGaussian"]


class DiagonalGaussian(Gaussian):
    def __init__(self, mean, diag):
        cov = Covariance.from_diagonal(diag)
        super().__init__(mean, cov)

    def update_diag(self, diag):
        cov = Covariance.from_diagonal(diag)
        super().update_cov(cov)


class IsotropicGaussian(DiagonalGaussian):
    def __init__(self, mean, std, size):
        diag = std**2 * np.ones(size)
        super().__init__(mean, diag)

    def update_std(self, std):
        diag = std**2 * np.ones(len(self))
        super().update_diag(diag)
