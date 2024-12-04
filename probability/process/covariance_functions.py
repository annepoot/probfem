import numpy as np

__all__ = [
    "CovarianceFunction",
    "SquaredExponential",
    "CovarianceFunctionSum",
    "CovarianceFunctionProduct",
]


class CovarianceFunction:
    def calc_cov(self, x1, x2):
        raise NotImplementedError("This has to be implemented in a child class")


class SquaredExponential(CovarianceFunction):
    def __init__(self, *, l, sigma):
        self.l = l
        self.sigma = sigma

    def calc_cov(self, x1, x2):
        if np.isscalar(x1):
            x1 = np.array([x1])
        if np.isscalar(x2):
            x2 = np.array([x2])

        X1 = np.tile(x1, (len(x2), 1))
        X2 = np.tile(x2, (len(x1), 1)).T

        dist = (X1 - X2) ** 2
        return self.sigma**2 * np.exp(-0.5 * dist / self.l**2)


class CovarianceFunctionSum(CovarianceFunction):
    def __init__(self, *covs):
        for cov in covs:
            if not isinstance(cov, CovarianceFunction):
                raise TypeError
        self.covs = covs

    def calc_cov(self, x1, x2):
        covsum = np.zeros((len(x1), len(x2)))
        for cov in self.covs:
            covsum += cov.calc_cov(x1, x2)
        return covsum


class CovarianceFunctionProduct(CovarianceFunction):
    def __init__(self, *covs):
        for cov in covs:
            if not isinstance(cov, CovarianceFunction):
                raise TypeError
        self.covs = covs

    def calc_cov(self, x1, x2):
        covprod = np.ones((len(x1), len(x2)))
        for cov in self.covs:
            covprod *= cov.calc_cov(x1, x2)
        return covprod
