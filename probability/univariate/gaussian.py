from scipy.stats import norm

from ..distribution import UnivariateDistribution

__all__ = ["Gaussian"]


class Gaussian(UnivariateDistribution):
    def __init__(self, mean, std):
        self.latent = norm(loc=mean, scale=std)

    def update_mean(self, mean):
        self.latent = norm(loc=mean, scale=self.latent.std())

    def update_std(self, std):
        self.latent = norm(loc=self.latent.mean(), scale=std)

    def calc_mean(self):
        return self.latent.mean()

    def calc_var(self):
        return self.latent.var()

    def calc_std(self):
        return self.latent.std()

    def calc_sample(self, seed):
        return self.latent.rvs(random_state=seed)

    def calc_samples(self, n, seed):
        return self.latent.rvs(size=n, random_state=seed)

    def calc_pdf(self, x):
        return self.latent.pdf(x)

    def calc_logpdf(self, x):
        return self.latent.logpdf(x)
