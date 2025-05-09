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
            x1 = np.array([[x1]])
        elif len(x1.shape) == 1:
            x1 = np.reshape(x1, (-1, 1))

        if np.isscalar(x2):
            x2 = np.array([[x2]])
        elif len(x2.shape) == 1:
            x2 = np.reshape(x2, (-1, 1))

        dist = np.zeros((x1.shape[0], x2.shape[0]))
        assert x1.shape[1] == x2.shape[1]

        for dim in range(x1.shape[1]):
            dist += np.subtract.outer(x1[:, dim], x2[:, dim]) ** 2

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
