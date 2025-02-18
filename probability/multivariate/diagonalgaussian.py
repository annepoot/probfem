import numpy as np
from scipy.sparse import diags_array

from .gaussian import Gaussian
from .covariance import SymbolicCovariance

from util.linalg import Matrix

__all__ = ["DiagonalGaussian", "IsotropicGaussian"]


class DiagonalGaussian(Gaussian):
    def __init__(self, mean, diag):
        expr = Matrix(diags_array(diag), name="D")
        super().__init__(mean, SymbolicCovariance(expr))

    def update_diag(self, diag):
        expr = Matrix(diags_array(diag), name="D")
        super().update_cov(SymbolicCovariance(expr))


class IsotropicGaussian(DiagonalGaussian):
    def __init__(self, mean, std, size):
        diag = std**2 * np.ones(size)
        super().__init__(mean, diag)
        self.cov.expr.name = "eI"

    def update_std(self, std):
        diag = std**2 * np.ones(len(self))
        super().update_diag(diag)
        self.cov.expr.name = "eI"
