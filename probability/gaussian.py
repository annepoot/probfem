import numpy as np
from warnings import warn

from .distribution import Distribution

__all__ = ["Gaussian", "LinTransGaussian", "LinSolveGaussian", "ConditionalGaussian"]


class GaussianLike(Distribution):
    """
    This class defines all core functionality for Gaussian distributions.
    All classes that implement some sort of Gaussian are derived from this class
    """

    def __len__(self):
        raise NotImplementedError("This has to be implemented in a child class")

    def __add__(self, shift):
        if np.isscalar(shift):
            return LinTransGaussian(self, 1, shift)
        elif isinstance(shift, np.ndarray):
            return LinTransGaussian(self, 1, shift)
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
            return LinTransGaussian(self, scale, 0)
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

    def calc_sqrtcov(self):
        raise NotImplementedError("This has to be implemented in a child class")

    def calc_std(self):
        L = self.calc_sqrtcov()
        var = np.zeros(self._len)
        for i in range(self._len):
            var[i] = L[i] @ L[i]
            if var[i] < 0:
                if np.isclose(var[i], 0):
                    var[i] = 0
                else:
                    raise ValueError("Large negative value encountered!")
        return np.sqrt(var)

    def calc_sample(self, seed):
        raise NotImplementedError("This has to be implemented in a child class")

    def calc_samples(self, n, seed):
        raise NotImplementedError("This has to be implemented in a child class")

    def condition_on(self, operator, measurements, noise):
        return ConditionalGaussian(self, operator, measurements, noise)


class Gaussian(GaussianLike):
    def __init__(self, mean, cov):
        self._len = self._calc_len(mean, cov)

        self.update_mean(mean)
        self.update_cov(cov)

    def __len__(self):
        return self._len

    def update_mean(self, mean):
        if mean is None:
            self._mean = np.zeros(self._len)
        elif np.isscalar(mean):
            if mean != 0.0:
                raise ValueError
            self._mean = np.zeros(self._len)
        else:
            if not isinstance(mean, np.ndarray):
                raise TypeError
            if len(mean.shape) != 1:
                raise ValueError
            self._mean = mean

    def update_cov(self, cov):
        if np.isscalar(cov):
            self._cov = cov * np.identity(self._len)
        else:
            if not isinstance(cov, np.ndarray):
                raise TypeError
            if len(cov.shape) == 1:
                if cov.shape[0] != self._len:
                    raise ValueError
                self._cov = np.diag(cov)
            elif len(cov.shape) == 2:
                if cov.shape[0] != self._len or cov.shape[1] != self._len:
                    raise ValueError
                self._cov = cov

        self._sqrtcov = np.linalg.cholesky(self._cov)
        self._logdet = 2 * np.sum(np.log(np.diagonal(self._sqrtcov)))
        if abs(self._logdet) < 100:
            assert np.isclose(self._logdet, np.log(np.linalg.det(self._cov)))

    def calc_mean(self):
        return self._mean

    def calc_cov(self):
        return self._cov

    def calc_sqrtcov(self):
        return self._sqrtcov

    def calc_std(self):
        return np.sqrt(np.diagonal(self._cov))

    def calc_sample(self, seed):
        rng = np.random.default_rng(seed)
        return self._mean + self._sqrtcov @ rng.standard_normal(self._len)

    def calc_samples(self, n, seed):
        rng = np.random.default_rng(seed)
        return np.tile(self._mean, (n, 1)).T + self._sqrtcov @ rng.standard_normal(
            (self._len, n)
        )

    def calc_pdf(self, x):
        return np.exp(self.evaluate_logpdf(x))

    def calc_logpdf(self, x):
        # TODO: improve efficiency via backsubstitution
        potential = (
            1 / 2 * (x - self._mean) @ np.linalg.solve(self._cov, (x - self._mean))
        )
        return (
            -self._len / 2 * np.log(2 * np.pi)
            - self._len / 2 * self._logdet
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


class LinTransGaussian(GaussianLike):
    """
    y is Gaussian and defined implicitly by y = A x + b (= scale * latent + shift)
    where x is Gaussian too
    """

    def __init__(self, latent, scale, shift):
        if not isinstance(latent, GaussianLike):
            raise TypeError()

        self._latent = latent

        if np.isscalar(scale):
            self._scale = np.identity(len(self._latent)) * scale
        else:
            self._scale = scale

        if np.isscalar(shift):
            self._shift = np.ones(len(self._latent)) * shift
        else:
            self._shift = shift

        # check compatibility
        self._len = self._scale.shape[0]
        if self._scale.shape[0] != self._shift.shape[0]:
            raise ValueError("scale and shift have incompatible sizes")
        if self._scale.shape[1] != len(self._latent):
            raise ValueError("scale and latent have incompatible sizes")

    def __len__(self):
        return self._len

    def __iadd__(self, shift):
        if np.isscalar(shift):
            self._shift += shift
            return self
        elif isinstance(shift, np.ndarray):
            self._shift += shift
            return self
        else:
            raise ValueError("cannot handle shift of type '{}'".format(type(shift)))

    def __imul__(self, scale):
        if np.isscalar(scale):
            self._scale *= scale
            return self
        else:
            raise ValueError("cannot handle scale of type '{}'".format(type(scale)))

    def calc_mean(self):
        return self._scale @ self._latent.calc_mean() + self._shift

    def calc_cov(self):
        return self._scale @ self._latent.calc_cov() @ self._scale.T

    def calc_sqrtcov(self):
        return self._scale @ self._latent.calc_sqrtcov()

    def calc_sample(self, seed):
        return self._scale @ self._latent.calc_sample(seed) + self._shift

    def calc_samples(self, n, seed):
        return (
            self._scale @ self._latent.calc_samples(n, seed)
            + np.tile(self._shift, (n, 1)).T
        )

    def to_gaussian(self):
        return Gaussian(self.calc_mean(), self.calc_cov())


class LinSolveGaussian(GaussianLike):
    """
    y is Gaussian and defined implicitly by A y = x (= inv^-1 * latent)
    where x is Gaussian too
    """

    def __init__(self, latent, inv, explicit=False):
        if not isinstance(latent, GaussianLike):
            raise TypeError()

        self._latent = latent
        self._inv = inv
        self._explicit = explicit

        if self._explicit:
            warn("Explicit inversion of fine-scale matrix!")
            self._explicitinv = np.linalg.inv(self._inv)

        # check compatibility
        self._len = self._inv.shape[0]
        if self._inv.shape[0] != len(self._latent):
            raise ValueError("inv and latent have incompatible sizes")
        if self._inv.shape[1] != len(self._latent):
            raise ValueError("inv and latent have incompatible sizes")

    def __len__(self):
        return self._len

    def calc_mean(self):
        return np.linalg.solve(self._inv, self._latent.calc_mean())

    def calc_cov(self):
        if self._explicit:
            return self._explicitinv @ self._latent.calc_cov() @ self._explicitinv.T
        else:
            raise ValueError(
                "Set explicit to True to explicitly compute the posterior covariance!"
            )

    def calc_sqrtcov(self):
        if self._explicit:
            return self._explicitinv @ self._latent.calc_sqrtcov()
        else:
            raise ValueError(
                "Set explicit to True to explicitly compute the posterior covariance!"
            )

    def calc_sample(self, seed):
        latent_sample = self._latent.calc_sample(seed)
        if self._explicit:
            return self._explicitinv @ latent_sample
        else:
            return np.linalg.solve(self._inv, latent_sample)

    def calc_samples(self, n, seed):
        latent_samples = self._latent.calc_samples(n, seed)
        if self._explicit:
            return self._explicitinv @ latent_samples
        else:
            return np.linalg.solve(self._inv, latent_samples)


class ConditionalGaussian(GaussianLike):
    """
    y is Gaussian and defined by conditioning on A x = b (linop x = obs)
    where x is Gaussian too
    """

    def __init__(self, latent, linop, obs, noise):
        if not isinstance(latent, GaussianLike):
            raise TypeError()

        self._latent = latent
        if np.isscalar(obs):
            self._linop = np.reshape(linop, (1, -1))
            self._obs = np.array([obs])
        else:
            self._linop = linop
            self._obs = obs
        self._noise = noise

        # check compatibility
        self._len = len(self._latent)
        if self._linop.shape[0] != self._obs.shape[0]:
            raise ValueError("operator and measurements have incompatible sizes")
        if self._linop.shape[1] != len(self._latent):
            raise ValueError("operator and latent have incompatible sizes")

        Sigma = self._latent.calc_cov()

        gram = self._linop @ Sigma @ self._linop.T

        if self._noise is None:
            # Compute the pseudoinverse, in case of linearly dependent observations
            tol = 1e-12

            if len(gram) == 1:
                if gram[0, 0] > tol:
                    graminv = 1 / gram
                elif gram[0, 0] >= -tol:
                    graminv = np.zeros((1, 1))
                else:
                    raise ValueError("Negative eigenvalue in Gram matrix!")

            else:
                lamb, Q = np.linalg.eigh(gram)
                for i, l in enumerate(lamb):
                    if l > tol:
                        lamb[i] = 1 / l
                    elif l >= -tol:
                        lamb[i] = 0
                    else:
                        raise ValueError("Negative eigenvalue in Gram matrix!")

                graminv = (Q * lamb) @ Q.T

        else:
            gram += np.identity(len(self._obs)) * self._noise
            graminv = np.linalg.inv(gram)

        self._kalgain = Sigma @ self._linop.T @ graminv

    def __len__(self):
        return self._len

    def calc_mean(self):
        mean = self._latent.calc_mean()
        return mean + self._kalgain @ (self._obs - self._linop @ mean)

    def calc_cov(self):
        Sigma = self._latent.calc_cov()
        return Sigma - self._kalgain @ self._linop @ Sigma

    def calc_sqrtcov(self):
        if self._noise is None:
            P = np.identity(self._len) - self._kalgain @ self._linop
            return P @ self._latent.calc_sqrtcov()
        else:
            raise ValueError(
                "Not sure how to handle calc_sqrtcov in case of observation noise"
            )

    def calc_sample(self, seed):
        sample = self._latent.calc_sample(seed)
        return sample + self._kalgain @ (self._obs - self._linop @ sample)

    def calc_samples(self, n, seed):
        samples = self._latent.calc_samples(n, seed)
        return samples + self._kalgain @ (
            np.tile(self._obs, (n, 1)).T - self._linop @ samples
        )
