import numpy as np

from probability import Likelihood
from probability.observation import ObservationOperator
from probability.multivariate import GaussianLike, IndependentGaussianSum
from probability.process.gaussian_process import GaussianProcess
from probability.process.covariance_functions import CovarianceFunction


__all__ = ["StatFEMLikelihood"]


class StatFEMLikelihood(Likelihood):
    def __init__(self, operator, values, rho, d, e, locations):
        assert isinstance(operator, ObservationOperator)
        self.operator = operator
        self.values = values

        assert np.isscalar(rho)
        self.rho = rho

        assert isinstance(d, GaussianProcess)
        assert isinstance(d.cov, CovarianceFunction)
        self.d = d

        assert isinstance(e, GaussianLike)
        self.e = e

        self.locations = locations

    def calc_pdf(self, x):
        prediction = self.operator.calc_prediction(x)
        de = IndependentGaussianSum(self.d.evaluate_at(self.locations), self.e)
        de = de.to_gaussian(allow_singular=True)
        de.update_mean(self.rho * prediction)
        return de.calc_pdf(self.values)

    def calc_logpdf(self, x):
        prediction = self.operator.calc_prediction(x)
        de = IndependentGaussianSum(self.d.evaluate_at(self.locations), self.e)
        de = de.to_gaussian(allow_singular=True)
        de.update_mean(self.rho * prediction)
        return de.calc_logpdf(self.values)
