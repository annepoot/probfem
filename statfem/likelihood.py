import numpy as np
from warnings import warn

from myjive.util.proputils import (
    split_key,
    get_attr_recursive,
    set_attr_recursive,
)

from probability import Likelihood
from probability.observation import ObservationOperator
from probability.multivariate.gaussian import GaussianLike
from probability.process.gaussian_process import GaussianProcess
from probability.process.covariance_functions import CovarianceFunction


__all__ = ["StatFEMLikelihood", "ParametrizedStatFEMLikelihood"]


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
        de = (self.d.evaluate_at(self.locations) + self.e).to_gaussian()
        de.update_mean(self.rho * prediction)
        return de.calc_pdf(self.values)

    def calc_logpdf(self, x):
        prediction = self.operator.calc_prediction(x)
        de = (self.d.evaluate_at(self.locations) + self.e).to_gaussian()
        de.update_mean(self.rho * prediction)
        return de.calc_logpdf(self.values)


class ParametrizedStatFEMLikelihood(StatFEMLikelihood):
    def __init__(self, operator, values, rho, d, e, locations, hyperparameters):
        super().__init__(
            operator=operator, values=values, rho=rho, d=d, e=e, locations=locations
        )
        self.hyperparameters = hyperparameters

    def calc_pdf(self, x):
        # split off hyperparameters from the back, and update them
        nhyp = len(self.hyperparameters)
        x_param, x_hyper = x[:-nhyp], x[-nhyp:]
        for key, value in zip(self.hyperparameters, x_hyper):
            keys = split_key(key)
            warn("Very dirty hack!")
            tvalue = np.exp(value)
            assert get_attr_recursive(self, keys) is not None
            set_attr_recursive(self, keys, tvalue)

        return super().calc_pdf(x_param)

    def calc_logpdf(self, x):
        # split off hyperparameters from the back, and update them
        nhyp = len(self.hyperparameters)
        x_param, x_hyper = x[:-nhyp], x[-nhyp:]
        for key, value in zip(self.hyperparameters, x_hyper):
            keys = split_key(key)
            warn("Very dirty hack!")
            tvalue = np.exp(value)
            assert get_attr_recursive(self, keys) is not None
            set_attr_recursive(self, keys, tvalue)

        return super().calc_logpdf(x_param)
