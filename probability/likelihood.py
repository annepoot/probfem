import numpy as np

from myjive.util.proputils import (
    split_key,
    get_attr_recursive,
    set_or_call_attr_recursive,
)

from probability.distribution import Distribution
from probability.observation import ObservationOperator


__all__ = [
    "Likelihood",
    "ParametrizedLikelihood",
    "ProportionalPosterior",
    "TemperedPosterior",
]


class Likelihood(Distribution):
    def __init__(self, operator, values, noise):
        assert isinstance(operator, ObservationOperator)
        self.operator = operator
        self.values = values

        assert isinstance(noise, Distribution)
        self.noise = noise

    def calc_pdf(self, x):
        prediction = self.operator.calc_prediction(x)
        if np.isnan(np.sum(prediction)):
            return -np.inf
        else:
            self.noise.update_mean(prediction)
            return self.noise.calc_pdf(self.values)

    def calc_logpdf(self, x):
        prediction = self.operator.calc_prediction(x)
        if np.isnan(np.sum(prediction)):
            return -np.inf
        else:
            self.noise.update_mean(prediction)
            return self.noise.calc_logpdf(self.values)


class ParametrizedLikelihood(Likelihood):
    def __init__(self, likelihood, hyperparameters):
        assert isinstance(likelihood, Likelihood)
        self.likelihood = likelihood
        self.hyperparameters = hyperparameters

    def calc_pdf(self, x):
        # split off hyperparameters from the back, and update them
        nhyp = len(self.hyperparameters)
        x_param, x_hyper = x[:-nhyp], x[-nhyp:]
        for key, value in zip(self.hyperparameters, x_hyper):
            keys = split_key(key)
            assert get_attr_recursive(self, keys) is not None
            set_or_call_attr_recursive(self.likelihood, keys, value)

        return self.likelihood.calc_pdf(x_param)

    def calc_logpdf(self, x):
        # split off hyperparameters from the back, and update them
        nhyp = len(self.hyperparameters)
        x_param, x_hyper = x[:-nhyp], x[-nhyp:]
        for key, value in zip(self.hyperparameters, x_hyper):
            keys = split_key(key)
            assert get_attr_recursive(self.likelihood, keys) is not None
            set_or_call_attr_recursive(self.likelihood, keys, value)

        return self.likelihood.calc_logpdf(x_param)


class ProportionalPosterior(Distribution):
    def __init__(self, prior, likelihood):
        assert isinstance(prior, Distribution)
        assert isinstance(likelihood, Likelihood)
        self.prior = prior
        self.likelihood = likelihood

    def __len__(self):
        return len(self.prior)

    def calc_pdf(self, x):
        prior_pdf = self.prior.calc_pdf(x)
        if prior_pdf == 0.0:
            return 0.0
        else:
            likelihood_pdf = self.likelihood.calc_pdf(x)
            return prior_pdf * likelihood_pdf

    def calc_logpdf(self, x):
        prior_logpdf = self.prior.calc_logpdf(x)
        if prior_logpdf < 0.0 and np.isinf(prior_logpdf):
            return prior_logpdf
        else:
            likelihood_logpdf = self.likelihood.calc_logpdf(x)
            return prior_logpdf + likelihood_logpdf


class TemperedPosterior(Distribution):
    def __init__(self, prior, likelihood):
        assert isinstance(prior, Distribution)
        assert isinstance(likelihood, Likelihood)
        self.prior = prior
        self.likelihood = likelihood

    def __len__(self):
        return len(self.prior)

    def set_temperature(self, temp):
        self._temp = temp

    def calc_pdf(self, x):
        prior_pdf = self.prior.calc_pdf(x)
        if prior_pdf == 0.0:
            return 0.0
        else:
            likelihood_pdf = self.likelihood.calc_pdf(x)
            return prior_pdf * likelihood_pdf**self._temp

    def calc_logpdf(self, x):
        prior_logpdf = self.prior.calc_logpdf(x)
        if prior_logpdf < 0.0 and np.isinf(prior_logpdf):
            return prior_logpdf
        else:
            likelihood_logpdf = self.likelihood.calc_logpdf(x)
            return prior_logpdf + self._temp * likelihood_logpdf
