import numpy as np
from warnings import warn

from probability import Likelihood
from probability.observation import ObservationOperator
from probability.multivariate import GaussianLike, IndependentGaussianSum
from probability.process.gaussian_process import GaussianProcess
from probability.process.covariance_functions import CovarianceFunction


__all__ = ["StatFEMLikelihood"]


class StatFEMLikelihood(Likelihood):
    def __init__(self, operator, values, rho, d, e):
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

    def calc_logpdf(self, x):
        try:
            prediction = self.operator.calc_prediction(x)
        except Exception as e:
            warn(f"exception caught: {e}\n\nreturning logpdf=-inf\n")
            return -np.inf

        if np.isnan(np.sum(prediction)):
            return -np.inf
        else:
            locations = self.operator.output_locations
            de = IndependentGaussianSum(self.d.evaluate_at(locations), self.e)
            de = de.to_gaussian(allow_singular=True)
            de.update_mean(self.rho * prediction)
            return de.calc_logpdf(self.values)
