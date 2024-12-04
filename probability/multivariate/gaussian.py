import numpy as np
from scipy.stats import multivariate_normal
from warnings import warn

from ..distribution import MultivariateDistribution

__all__ = ["Gaussian", "ScaledGaussian", "ShiftedGaussian", "ConditionedGaussian"]


class GaussianLike(MultivariateDistribution):
    """
    This class defines all core functionality for Gaussian distributions.
    All classes that implement some sort of Gaussian are derived from this class
    """

    def __len__(self):
        raise NotImplementedError("This has to be implemented in a child class")

    def __add__(self, shift):
        if isinstance(shift, np.ndarray):
            return ShiftedGaussian(self, shift)
        else:
            raise ValueError("cannot handle shift of type '{}'".format(type(shift)))

    def __radd__(self, shift):
        return self.__add__(shift)

    def __iadd__(self, shift):
        return self.__add__(shift)

    def __sub__(self, shift):
        return self.__add__(-shift)

    def __rsub__(self, shift):
        return self.__mul__(-1).__add__(shift)

    def __isub__(self, shift):
        return self.__iadd__(-shift)

    def __mul__(self, scale):
        if np.isscalar(scale):
            return ScaledGaussian(self, scale)
        else:
            raise ValueError("cannot handle scale of type '{}'".format(type(scale)))

    def __rmul__(self, scale):
        return self.__mul__(scale)

    def __imul__(self, scale):
        return self.__mul__(scale)

    def __truediv__(self, scale):
        return self.__mul__(1 / scale)

    def calc_mean(self):
        raise NotImplementedError("This has to be implemented in a child class")

    def calc_cov(self):
        raise NotImplementedError("This has to be implemented in a child class")

    def calc_diag(self):
        return np.diagonal(self.calc_cov())

    def calc_std(self):
        return np.sqrt(self.calc_diag())

    def calc_sample(self, seed):
        raise NotImplementedError("This has to be implemented in a child class")

    def calc_samples(self, n, seed):
        raise NotImplementedError("This has to be implemented in a child class")

    def condition_on(self, operator, measurements):
        return ConditionedGaussian(self, operator, measurements)

    def to_gaussian(self):
        return Gaussian(self.calc_mean(), self.calc_cov())


class Gaussian(GaussianLike):
    def __init__(self, mean, cov, allow_singular=False):
        self.latent = multivariate_normal(
            mean=mean, cov=cov, allow_singular=allow_singular
        )

    def __len__(self):
        return len(self.latent.mean)

    def update_mean(self, mean):
        self.latent = multivariate_normal(mean, self.latent.cov)

    def update_cov(self, cov):
        self.latent = multivariate_normal(self.latent.mean, cov)

    def calc_mean(self):
        return self.latent.mean

    def calc_cov(self):
        return self.latent.cov

    def calc_diag(self):
        return np.diagonal(self.latent.cov)

    def calc_std(self):
        return np.sqrt(np.diagonal(self.latent.cov))

    def calc_sample(self, seed):
        return self.latent.rvs(random_state=seed).flatten()

    def calc_samples(self, n, seed):
        return self.latent.rvs(size=n, random_state=seed)

    def calc_pdf(self, x):
        return self.latent.pdf(x)

    def calc_logpdf(self, x):
        return self.latent.logpdf(x)


class ScaledGaussian(GaussianLike):

    def __init__(self, latent, scale):
        assert isinstance(latent, Gaussian)
        self.latent = latent

        if np.isscalar(scale):
            self.scale = scale * np.identity(len(self.latent))
        else:
            assert isinstance(scale, np.ndarray)
            assert len(scale.shape) == 2
            assert scale.shape[1] == len(self.latent)
            self.scale = scale

    def __len__(self):
        return self.scale.shape[0]

    def calc_mean(self):
        return self.scale @ self.latent.calc_mean()

    def calc_cov(self):
        return self.scale @ self.latent.calc_cov() @ self.scale.T

    def calc_sample(self, seed):
        return self.scale @ self.latent.calc_sample(seed)

    def calc_samples(self, n, seed):
        return self.latent.calc_samples(n, seed) @ self.scale.T


class ShiftedGaussian(GaussianLike):

    def __init__(self, latent, shift):
        assert isinstance(latent, Gaussian)
        self.latent = latent

        assert isinstance(shift, np.ndarray)
        assert len(shift.shape) == 1
        assert shift.shape[0] == len(self.latent)
        self.shift = shift

    def __len__(self):
        return len(self.latent)

    def calc_mean(self):
        return self.latent.calc_mean() + self.shift

    def calc_cov(self):
        return self.latent.calc_cov()

    def calc_sample(self, seed):
        return self.latent.calc_sample(seed) + self.shift

    def calc_samples(self, n, seed):
        return self.latent.calc_samples(n, seed) + np.tile(self.shift, (n, 1))


class ConditionedGaussian(GaussianLike):

    def __init__(self, prior, linop, obs):
        assert isinstance(prior, GaussianLike)
        self.prior = prior

        assert isinstance(linop, np.ndarray)
        if len(linop.shape) == 1:
            linop = np.reshape(linop, (1, -1))
        assert len(linop.shape) == 2
        assert linop.shape[1] == len(self.prior)
        self.linop = linop

        if np.isscalar(obs):
            obs = np.array([obs])
        assert isinstance(obs, np.ndarray)
        assert len(obs.shape) == 1
        assert obs.shape[0] == linop.shape[0]
        self.obs = obs

        prior_cov = self.prior.calc_cov()
        inv_gram = np.linalg.pinv(self.linop @ prior_cov @ self.linop.T)
        self.kalman_gain = prior_cov @ self.linop.T @ inv_gram

    def __len__(self):
        return len(self.prior)

    def calc_mean(self):
        prior_mean = self.prior.calc_mean()
        return prior_mean + self.kalman_gain @ (self.obs - self.linop @ prior_mean)

    def calc_cov(self):
        Q = np.identity(len(self.prior)) - self.kalman_gain @ self.linop
        return Q @ self.prior.calc_cov() @ Q.T

    def calc_sample(self, seed):
        raise NotImplementedError("Not tested yet!")
        sample = self.prior.calc_sample(seed)
        return sample + self.kalman_gain @ (self.obs - self.linop @ sample)

    def calc_samples(self, n, seed):
        samples = self.prior.calc_samples(n, seed)
        return (
            samples
            + (np.tile(self.obs, (n, 1)) - samples @ self.linop.T) @ self.kalman_gain.T
        )
