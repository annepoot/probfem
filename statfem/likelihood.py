import numpy as np
from warnings import warn

from myjive.util.proputils import split_off_type, split_key, set_attr_recursive

from probability import Likelihood
from probability.observation import ObservationOperator
from probability.multivariate.gaussian import GaussianLike
from probability.process.gaussian_process import GaussianProcess
from probability.process.covariance_functions import CovarianceFunction


__all__ = ["StatFEMLikelihood", "ParametrizedStatFEMLikelihood"]


class StatFEMLikelihood(Likelihood):
    def __init__(self, operator, values, rho, d, e, locations):
        operator_cls, operator_kws = split_off_type(operator)

        if not issubclass(operator_cls, ObservationOperator):
            raise TypeError

        self.operator = operator_cls(**operator_kws)
        self.values = values

        assert np.isscalar(rho)
        self.rho = rho

        misspec_cls, misspec_kws = split_off_type(d)
        assert issubclass(misspec_cls, GaussianProcess)
        dcov_cls, dcov_kws = split_off_type(misspec_kws["cov"])
        assert issubclass(dcov_cls, CovarianceFunction)
        dcov = dcov_cls(**dcov_kws)
        self.d = misspec_cls(mean=None, cov=dcov)

        noise_cls, noise_kws = split_off_type(e)
        assert issubclass(noise_cls, GaussianLike)
        self.e = noise_cls(**noise_kws)

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
            set_attr_recursive(self, keys, tvalue)

        return super().calc_logpdf(x_param)
