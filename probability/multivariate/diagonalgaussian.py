import numpy as np
from warnings import warn

from .gaussian import GaussianLike

__all__ = ["DiagonalGaussian", "IsotropicGaussian"]


class DiagonalGaussian(GaussianLike):
    def __init__(self, mean, diag):
        self._len = self._calc_len(mean, diag)

        self.update_mean(mean)
        self.update_diag(diag)

    def __len__(self):
        return self._len

    def update_mean(self, mean):
        if mean is None:
            self.mean = np.zeros(self._len)
        elif np.isscalar(mean):
            if mean != 0.0:
                raise ValueError
            self.mean = np.zeros(self._len)
        else:
            if not isinstance(mean, np.ndarray):
                raise TypeError
            if len(mean.shape) != 1:
                raise ValueError
            if mean.shape[0] != self._len:
                raise ValueError
            self.mean = mean

    def update_diag(self, diag):
        if np.isscalar(diag):
            self.diag = diag * np.ones(self._len)
        else:
            if not isinstance(diag, np.ndarray):
                raise TypeError
            if len(diag.shape) != 1:
                raise ValueError
            if diag.shape[0] != self._len:
                raise ValueError
            self.diag = diag

    def calc_mean(self):
        return self.mean

    def calc_cov(self):
        return np.diag(self.diag)

    def calc_sqrtcov(self):
        return np.diag(self.calc_std())

    def calc_std(self):
        return np.sqrt(self.diag)

    def calc_sample(self, seed):
        raise NotImplementedError("Not tested yet!")
        rng = np.random.default_rng(seed)
        return self.mean + self.calc_std() * rng.standard_normal(self._len)

    def calc_samples(self, n, seed):
        raise NotImplementedError("Not tested yet!")
        rng = np.random.default_rng(seed)
        return (
            np.tile(self.mean, (n, 1)).T
            + (self.calc_std() * rng.standard_normal((self._len, n)).T).T
        )

    def calc_pdf(self, x):
        return np.exp(self.calc_logpdf(x))

    def calc_logpdf(self, x):
        # TODO: improve efficiency via backsubstitution
        potential = 0.5 * (x - self.mean) @ ((x - self.mean) / self.diag)

        return (
            -0.5 * self._len * np.log(2 * np.pi)
            - 0.5 * np.sum(np.log(self.diag))
            - potential
        )

    def _calc_len(self, mean, cov):
        if mean is None:
            if np.isscalar(cov):
                return 1
            else:
                return len(cov)
        else:
            if np.isscalar(mean):
                return 1
            else:
                return len(mean)


class IsotropicGaussian(DiagonalGaussian):
    def __init__(self, mean, std):
        self._len = len(mean)
        self.update_mean(mean)
        self.update_std(std)

    def __len__(self):
        return self._len

    def update_mean(self, mean):
        if mean is None:
            self.mean = np.zeros(self._len)
        elif np.isscalar(mean):
            self.mean = mean * np.zeros(self._len)
        else:
            if not isinstance(mean, np.ndarray):
                raise TypeError
            if len(mean.shape) != 1:
                raise ValueError
            if mean.shape[0] != self._len:
                raise ValueError
            self.mean = mean

    def update_std(self, std):
        if np.isscalar(std):
            if std < 0:
                raise ValueError
            self.std = std
        else:
            raise ValueError

    def calc_mean(self):
        return self.mean

    def calc_cov(self):
        return self.std**2 * np.identity(self._len)

    def calc_sqrtcov(self):
        return self.std * np.identity(self._len)

    def calc_std(self):
        return self.std * np.ones(self._len)

    def calc_sample(self, seed):
        rng = np.random.default_rng(seed)
        return self.mean + self.std * rng.standard_normal(self._len)

    def calc_samples(self, n, seed):
        raise NotImplementedError("Not tested yet!")
        rng = np.random.default_rng(seed)
        return np.tile(self.mean, (n, 1)).T + self.std * rng.standard_normal(
            (self._len, n)
        )

    def calc_pdf(self, x):
        return np.exp(self.calc_logpdf(x))

    def calc_logpdf(self, x):
        potential = 0.5 * (x - self.mean) @ (x - self.mean) / self.std**2

        return (
            -self._len / 2 * np.log(2 * np.pi)
            - self._len * np.log(self.std)
            - potential
        )

    def _calc_len(self, mean, std):
        if mean is None:
            return 1
        elif np.isscalar(mean):
            return 1
        else:
            return len(mean)
