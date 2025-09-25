import numpy as np
from scipy.special import logsumexp
import pandas as pd

from ..distribution import MultivariateDistribution
from .gaussian import Gaussian


class Mixture(MultivariateDistribution):

    def __init__(self, distributions, weights=None):
        self.distributions = distributions

        if weights is None:
            self.weights = np.ones(len(self.distributions))
        else:
            self.weights = weights

        self.weights /= np.sum(self.weights)
        assert len(self.distributions) == len(self.weights)

    def calc_sample(self, rng):
        dist = rng.choice(self.distributions, p=self.weights)
        return dist.calc_sample(rng)

    def calc_samples(self, n, rng):
        samples = [None] * n
        for i in range(n):
            samples[i] = self.calc_sample(rng)
        return np.array(samples)

    def calc_pdf(self, x):
        pdfs = [None] * len(self.distributions)
        for i, (dist, weight) in enumerate(zip(self.distributions, self.weights)):
            pdfs[i] = weight * dist.calc_pdf(x)
        return np.sum(pdfs, axis=0)

    def calc_logpdf(self, x):
        logpdfs = [None] * len(self.distributions)
        for i, (dist, weight) in enumerate(zip(self.distributions, self.weights)):
            logpdfs[i] = np.log(weight) + dist.calc_logpdf(x)
        return logsumexp(logpdfs, axis=0)


class EmpiricalMixture(MultivariateDistribution):

    def __init__(self, data, covariance):
        df = pd.DataFrame(data)
        condensed = df.groupby(df.columns.tolist(), as_index=False).size()
        self.data = condensed[df.columns].to_numpy()
        self.weights = condensed["size"].to_numpy() / np.sum(condensed["size"])
        self.covariance = covariance
        self.precision = np.linalg.inv(self.covariance)
        self.dist = Gaussian(None, self.covariance)

        assert len(self.data) == len(self.weights)

    def calc_sample(self, rng):
        idx = rng.choice(np.arange(len(self.data)), p=self.weights)
        return self.data[idx] + self.dist.calc_sample(rng)

    def calc_samples(self, n, rng):
        idx = rng.choice(np.arange(len(self.data)), size=n, p=self.weights)
        return self.data[idx] + self.dist.calc_samples(n, rng)

    def calc_pdf(self, x):
        # # for sure correct
        # pdf = 0.0
        # for mean, weight in zip(self.data, self.weights):
        #     d = mean - x
        #     pdf += weight * self.dist.calc_pdf(d)

        # quicker
        return np.exp(self.calc_logpdf(x))

    def calc_logpdf(self, x):
        # exploit the fact that the covariance is the same for each point
        ds = self.data - x
        mahals = np.sum(ds @ self.precision * ds, axis=1)
        logc = np.log(self.dist.calc_pdf(np.zeros_like(x)))
        return logc + logsumexp(np.log(self.weights) - 0.5 * mahals)
