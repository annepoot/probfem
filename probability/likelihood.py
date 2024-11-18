from myjive.util.proputils import split_off_type

from probability.distribution import Distribution
from probability.observation import ObservationOperator


__all__ = ["Likelihood", "ProportionalPosterior"]


class Likelihood(Distribution):

    def __init__(self, operator, values, noise):

        operator_cls, operator_kws = split_off_type(operator)

        if not issubclass(operator_cls, ObservationOperator):
            raise TypeError

        self._operator = operator_cls(**operator_kws)
        self._values = values

        noise_cls, noise_kws = split_off_type(noise)
        assert issubclass(noise_cls, Distribution)

        self._noise = noise_cls(**noise_kws)

    def calc_pdf(self, x):
        prediction = self._operator.calc_prediction(x)
        self._noise.set_mean(prediction)
        return self._noise.calc_pdf(self._values)

    def calc_logpdf(self, x):
        prediction = self._operator.calc_prediction(x)
        self._noise.update_mean(prediction)
        return self._noise.calc_logpdf(self._values)


class ProportionalPosterior(Distribution):

    def __init__(self, prior, likelihood):
        prior_cls, prior_kws = split_off_type(prior)
        likelihood_cls, likelihood_kws = split_off_type(likelihood)

        assert issubclass(prior_cls, Distribution)
        assert issubclass(likelihood_cls, Likelihood)

        self._prior = prior_cls(**prior_kws)
        self._likelihood = likelihood_cls(**likelihood_kws)

    def __len__(self):
        return len(self._prior)

    def calc_pdf(self, x):
        prior_pdf = self._prior.calc_pdf(x)
        likelihood_pdf = self._likelihood.calc_pdf(x)
        return prior_pdf + likelihood_pdf

    def calc_logpdf(self, x):
        prior_logpdf = self._prior.calc_logpdf(x)
        likelihood_logpdf = self._likelihood.calc_logpdf(x)
        return prior_logpdf + likelihood_logpdf
