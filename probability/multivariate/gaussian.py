import numpy as np
from scipy.sparse import issparse, eye_array
from scipy.sparse.linalg import spsolve
from scipy.stats import multivariate_normal
from warnings import warn
import sksparse.cholmod as cm

from util.linalg import Matrix, MatMulChain
from ..distribution import MultivariateDistribution
from .covariance import Covariance, SymbolicCovariance


__all__ = [
    "GaussianLike",
    "Gaussian",
    "ScaledGaussian",
    "ShiftedGaussian",
    "ConditionedGaussian",
    "IndependentGaussianSum",
]


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
        elif isinstance(shift, GaussianLike):
            warn("assuming independence between added Gaussians")
            return IndependentGaussianSum(self, shift)
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

    def __matmul__(self, scale):
        if issparse(scale) or isinstance(scale, np.ndarray):
            return ScaledGaussian(self, scale.T)
        else:
            raise ValueError("cannot handle matmul of type '{}'".format(type(scale)))

    def __truediv__(self, scale):
        return self.__mul__(1 / scale)

    def calc_mean(self):
        raise NotImplementedError("This has to be implemented in a child class")

    def get_cov(self):
        return SymbolicCovariance(Matrix(self.calc_cov(), name="C"))

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

    def calc_pdf(self, x):
        return np.exp(self.calc_logpdf(x))

    def calc_logpdf(self, x):
        return (
            -0.5 * len(self) * np.log(2 * np.pi)
            - 0.5 * self.calc_logpdet()
            - 0.5 * self.calc_mahal_squared(x)
        )

    def calc_mahal(self, x):
        return np.sqrt(self.calc_mahal_squared(x))

    def calc_mahal_squared(self, x):
        raise NotImplementedError("This has to be implemented in a child class")

    def calc_pdet(self):
        return np.exp(self.calc_logpdet())

    def calc_logpdet(self):
        raise NotImplementedError("This has to be implemented in a child class")

    def condition_on(self, operator, measurements):
        return ConditionedGaussian(self, operator, measurements)

    def to_gaussian(self, allow_singular=False):
        cov = self.get_cov()
        use_scipy_latent = not isinstance(cov, (Covariance, SymbolicCovariance))
        return Gaussian(
            self.calc_mean(),
            cov,
            allow_singular=allow_singular,
            use_scipy_latent=use_scipy_latent,
        )


class Gaussian(GaussianLike):
    def __init__(self, mean, cov, allow_singular=False, use_scipy_latent=True):
        self.allow_singular = allow_singular

        if isinstance(cov, SymbolicCovariance):
            self.use_scipy_latent = False
        else:
            self.use_scipy_latent = use_scipy_latent

        if self.use_scipy_latent:
            self.latent = multivariate_normal(
                mean=mean, cov=cov, allow_singular=self.allow_singular
            )
        else:
            self._len = self._get_len(mean, cov)
            self.update_mean(mean)
            self.update_cov(cov)

    def __len__(self):
        if self.use_scipy_latent:
            return len(self.latent.mean)
        else:
            return len(self.mean)

    def update_mean(self, mean):
        if self.use_scipy_latent:
            self.latent = multivariate_normal(
                mean, self.latent.cov, allow_singular=self.allow_singular
            )
        else:
            if mean is None:
                self.mean = np.zeros(self._len)
            else:
                assert len(mean.shape) == 1
                assert mean.shape[0] == self._len
                self.mean = mean

    def update_cov(self, cov):
        if self.use_scipy_latent:
            self.latent = multivariate_normal(
                self.latent.mean, cov, allow_singular=self.allow_singular
            )
        else:
            if isinstance(cov, SymbolicCovariance):
                self.cov = cov
            else:
                assert len(cov.shape) == 2
                assert cov.shape[0] == cov.shape[1] == self._len
                self.cov = SymbolicCovariance(Matrix(cov, name="C"))

    def calc_mean(self):
        if self.use_scipy_latent:
            return self.latent.mean
        else:
            return self.mean

    def get_cov(self):
        if self.use_scipy_latent:
            return self.latent.cov
        else:
            return self.cov

    def calc_cov(self):
        if self.use_scipy_latent:
            return self.latent.cov
        else:
            return self.cov.calc_cov()

    def calc_diag(self):
        return self.calc_cov().diagonal()

    def calc_std(self):
        return np.sqrt(self.calc_diag())

    def calc_sample(self, seed):
        if self.use_scipy_latent:
            return self.latent.rvs(random_state=seed).flatten()
        else:
            return self.mean + self.cov.calc_sample(seed)

    def calc_samples(self, n, seed):
        if self.use_scipy_latent:
            return self.latent.rvs(size=n, random_state=seed)
        else:
            return self.mean + self.cov.calc_samples(n, seed)

    def calc_mahal_squared(self, x):
        return self.cov.calc_mahal_squared(x - self.mean)

    def calc_logpdet(self):
        return self.cov.calc_logdet()

    def calc_pdf(self, x):
        if self.use_scipy_latent:
            return self.latent.pdf(x)
        else:
            logpdf = self.calc_logpdf(x)
            if logpdf <= 0.0:
                return -np.inf
            else:
                return np.exp(logpdf)

    def calc_logpdf(self, x):
        if self.use_scipy_latent:
            return self.latent.logpdf(x)
        else:
            return (
                -0.5 * len(self) * np.log(2 * np.pi)
                - 0.5 * self.calc_logpdet()
                - 0.5 * self.calc_mahal_squared(x)
            )

    def _get_len(self, mean, cov):
        if mean is None:
            return cov.shape[0]
        elif cov is None:
            return mean.shape[0]
        else:
            assert mean.shape[0] == cov.shape[0] == cov.shape[1]
            return mean.shape[0]


class ScaledGaussian(GaussianLike):

    def __init__(self, latent, scale):
        assert isinstance(latent, GaussianLike)
        self.latent = latent

        if np.isscalar(scale):
            self.scale = Matrix(scale * eye_array(len(self.latent)), name="aI")
        else:
            assert issparse(scale) or isinstance(scale, np.ndarray)
            assert len(scale.shape) == 2
            assert scale.shape[1] == len(self.latent)
            self.scale = Matrix(scale, name="A")

    def __len__(self):
        return self.scale.shape[0]

    def calc_mean(self):
        return self.scale @ self.latent.calc_mean()

    def calc_cov(self):
        return self.scale @ (self.scale @ self.latent.calc_cov()).T

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

        assert issparse(linop) or isinstance(linop, (np.ndarray, Matrix, MatMulChain))
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

        self.prior_cov = self.prior.get_cov()
        self.gram = Matrix(self.prior_cov.calc_gram(self.linop), name="G")
        self.kalman_gain = self.prior_cov @ self.linop.T @ self.gram.inv
        self.kalman_gain.simplify()

    def __len__(self):
        return len(self.prior)

    def calc_mean(self):
        prior_mean = self.prior.calc_mean()
        correction = self.kalman_gain @ (self.obs - self.linop @ prior_mean)
        return prior_mean + correction

    def calc_cov(self):
        correction = self.kalman_gain @ self.linop @ self.prior_cov.expr
        correction.simplify()

        warn("explicit covariance computation")
        return self.prior_cov.calc_cov() - correction.evaluate()

    def calc_sample(self, seed):
        sample = self.prior.calc_sample(seed)
        correction = self.kalman_gain @ (self.obs - self.linop @ sample)
        return sample + correction

    def calc_samples(self, n, seed):
        samples = self.prior.calc_samples(n, seed)
        obsmat = np.tile(np.array([self.obs]).T, (1, n))
        correction = (self.kalman_gain @ (obsmat - self.linop @ samples.T)).T
        return samples + correction


class IndependentGaussianSum(GaussianLike):

    def __init__(self, *gaussians):
        for gaussian in gaussians:
            assert isinstance(gaussian, GaussianLike)
            assert len(gaussian) == len(gaussians[0])

        self.gaussians = gaussians

    def __len__(self):
        return len(self.gaussians[0])

    def calc_mean(self):
        mean = np.zeros(len(self))
        for gaussian in self.gaussians:
            mean += gaussian.calc_mean()
        return mean

    def calc_cov(self):
        cov = np.zeros((len(self), len(self)))
        for gaussian in self.gaussians:
            cov += gaussian.calc_cov()
        return cov

    def calc_sample(self, seed):
        sample = np.zeros(len(self))
        rng = np.random.default_rng(seed)
        for gaussian in self.gaussians:
            sample += gaussian.calc_sample(rng)
        return sample

    def calc_samples(self, n, seed):
        samples = np.zeros((n, len(self)))
        rng = np.random.default_rng(seed)
        for gaussian in self.gaussians:
            samples += gaussian.calc_samples(n, rng)
        return samples

    def calc_logpdet(self):
        l, Q = self._calc_eigh()
        return np.sum(np.log(l))

    def calc_mahal_squared(self, x):
        l, Q = self._calc_eigh()
        d = self.calc_mean() - x
        d_eigh = Q.T @ d
        return d_eigh @ (d_eigh / l)

    def _calc_eigh(self):
        if not hasattr(self, "_eigh"):
            assert len(self.gaussians) == 2
            base = self.gaussians[0]
            noise = self.gaussians[1]

            assert isinstance(base, ScaledGaussian)
            assert isinstance(base.latent, ConditionedGaussian)

            scale = base.scale
            posterior = base.latent
            Sigma = posterior.prior.cov.expr
            linop = posterior.linop

            assert noise.cov.expr.is_diagonal
            eI = noise.cov.expr

            assert isinstance(Sigma, MatMulChain)
            assert len(Sigma) == 1
            downdate = Sigma @ linop.T @ posterior.gram.inv @ linop @ Sigma
            downdate.simplify()

            A_prior_At = scale @ Sigma @ scale.T
            A_downdate_At = scale @ downdate @ scale.T

            A_prior_At.evaluate()
            A_downdate_At.evaluate()

            full_cov = A_prior_At.evaluate() - A_downdate_At.evaluate() + eI.evaluate()

            self._eigh = np.linalg.eigh(full_cov)

        return self._eigh
