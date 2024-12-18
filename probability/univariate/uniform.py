from scipy.stats import uniform

from ..distribution import UnivariateDistribution

__all__ = ["Uniform"]


class Uniform(UnivariateDistribution):
    def __init__(self, a, b):
        self.latent = uniform(loc=a, scale=b - a)

    def update_mean(self, mean):
        width = self.calc_width()
        loc = mean - 0.5 * width
        self.latent = uniform(loc=loc, scale=width)

    def update_width(self, width):
        self.latent = uniform(loc=self.calc_mean(), scale=width)

    def calc_mean(self):
        return self.latent.mean()

    def calc_width(self):
        return self.latent.kwds["scale"]

    def calc_sample(self, seed):
        return self.latent.rvs(random_state=seed)

    def calc_samples(self, n, seed):
        return self.latent.rvs(size=n, random_state=seed)

    def calc_pdf(self, x):
        return self.latent.pdf(x)

    def calc_logpdf(self, x):
        return self.latent.logpdf(x)
