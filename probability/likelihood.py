import numpy as np
from warnings import warn

from myjive.util.proputils import split_key, get_attr_recursive, set_attr_recursive

from probability.distribution import Distribution
from probability.observation import ObservationOperator


__all__ = ["Likelihood", "ParametrizedLikelihood", "ProportionalPosterior"]


class Likelihood(Distribution):
    def __init__(self, operator, values, noise):
        assert isinstance(operator, ObservationOperator)
        self.operator = operator
        self.values = values

        assert isinstance(noise, Distribution)
        self.noise = noise

    def calc_pdf(self, x):
        prediction = self.operator.calc_prediction(x)
        self.noise.update_mean(prediction)
        return self.noise.calc_pdf(self.values)

    def calc_logpdf(self, x):
        prediction = self.operator.calc_prediction(x)
        self.noise.update_mean(prediction)
        return self.noise.calc_logpdf(self.values)


class ParametrizedLikelihood(Likelihood):
    def __init__(self, operator, values, noise, hyperparameters):
        super().__init__(operator=operator, values=values, noise=noise)
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
        likelihood_pdf = self.likelihood.calc_pdf(x)
        return prior_pdf * likelihood_pdf

    def calc_logpdf(self, x):
        prior_logpdf = self.prior.calc_logpdf(x)
        likelihood_logpdf = self.likelihood.calc_logpdf(x)
        return prior_logpdf + likelihood_logpdf
