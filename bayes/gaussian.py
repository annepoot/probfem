import numpy as np


class GaussianLike:
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

    def calc_std(self):
        raise NotImplementedError("This has to be implemented in a child class")

    def calc_samples(self):
        raise NotImplementedError("This has to be implemented in a child class")

    def condition_on(self, operator, measurements, noise):
        return ConditionalGaussian(self, operator, measurements, noise)


class DirectGaussian(GaussianLike):
    def __init__(self, mean, cov):
        if np.isscalar(mean):
            if mean == 0.0:
                self._mean = np.zeros(cov.shape[0])
            else:
                raise ValueError
        else:
            self._mean = mean
        self._cov = cov
        self._sqrtcov = np.linalg.cholesky(cov)

        # check compatibility
        self._len = len(self._mean)
        if self._cov.shape[0] != self._len or self._cov.shape[1] != self._len:
            raise ValueError("mean and cov have incompatible sizes")

    def __len__(self):
        return self._len

    def calc_mean(self):
        return self._mean

    def calc_cov(self):
        return self._cov

    def calc_std(self):
        return np.sqrt(np.diagonal(self._cov))

    def calc_samples(self):
        return self._mean + self._sqrtcov @ np.random.randn(self._n)


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

    def calc_samples(self):
        return self._scale @ self._latent.calc_samples() + self._shift

    def to_direct_gaussian(self):
        return DirectGaussian(self.calc_mean(), self.calc_cov())


class LinSolveGaussian(GaussianLike):
    """
    y is Gaussian and defined implicitly by A y = x (= inv^-1 * latent)
    where x is Gaussian too
    """

    def __init__(self, latent, inv):
        if not isinstance(latent, GaussianLike):
            raise TypeError()

        self._latent = latent
        self._inv = inv

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
        raise NotImplementedError("requires explicit inversion of inv!")

    def calc_samples(self):
        return np.linalg.solve(self._inv, self._latent.calc_samples())


class ConditionalGaussian(GaussianLike):
    """
    y is Gaussian and defined by conditioning on A x = b (linop x = obs)
    where x is Gaussian too
    """

    def __init__(self, latent, linop, obs, noise):
        if not isinstance(latent, GaussianLike):
            raise TypeError()

        self._latent = latent
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

        self._gram = self._linop @ Sigma @ self._linop.T
        self._gram += np.identity(len(self._obs)) * noise
        self._sqrtgram = np.linalg.cholesky(self._gram)
        self._kalgain = Sigma @ self._linop.T @ np.linalg.inv(self._gram)

    def __len__(self):
        return self._len

    def calc_mean(self):
        mean = self._latent.calc_mean()
        return mean - self._kalgain @ (self._linop @ mean - self._obs)

    def calc_cov(self):
        Sigma = self._latent.calc_cov()
        return Sigma - self._kalgain @ self._linop @ Sigma

    def calc_samples(self):
        sample = self._latent.calc_samples()
        return sample - self._kalgain @ (self._linop @ sample)
