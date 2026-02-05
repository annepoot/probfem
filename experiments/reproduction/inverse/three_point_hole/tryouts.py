import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

from probability.sampling import IndependenceSampler
from probability.distribution import UnivariateDistribution

from probability.univariate import Gaussian
from probability.multivariate import Mixture


class ArbitraryTarget(UnivariateDistribution):

    def __init__(self):
        self.mu_1 = -4
        self.mu_2 = 4
        self.sigma_1 = 2.0
        self.sigma_2 = 1.0
        self.w_1 = 1.0
        self.w_2 = 3.0

        self.A = 1.0
        self.A = quad(self.calc_pdf, -10, 10)[0]

    def calc_pdf(self, x):
        pdf_1 = np.exp(-((x - self.mu_1) ** 4) / self.sigma_1**4)
        pdf_2 = np.exp(-((x - self.mu_2) ** 4) / self.sigma_2**4)
        return (self.w_1 * pdf_1 + self.w_2 * pdf_2) / self.A

    def calc_logpdf(self, x):
        return np.log(self.calc_pdf(x))


class GaussianMixture(UnivariateDistribution):

    def __init__(self):
        self.mu_1 = -4
        self.mu_2 = 4
        self.sigma_1 = 1.0
        self.sigma_2 = 1.0
        self.w_1 = 1.0
        self.w_2 = 1.0

        self.p_1 = Gaussian(self.mu_1, self.sigma_1)
        self.p_2 = Gaussian(self.mu_2, self.sigma_2)

    def calc_sample(self, rng):
        if rng.uniform() < self.w_1 / (self.w_1 + self.w_2):
            return self.p_1.calc_sample(rng)
        else:
            return self.p_2.calc_sample(rng)

    def calc_pdf(self, x):
        pdf_1 = self.p_1.calc_pdf(x)
        pdf_2 = self.p_2.calc_pdf(x)
        return (self.w_1 * pdf_1 + self.w_2 * pdf_2) / (self.w_1 + self.w_2)

    def calc_logpdf(self, x):
        return np.log(self.calc_pdf(x))


# class IndependenceSampler:
#     def __init__(
#         self,
#         *,
#         target,
#         proposal,
#         n_sample,
#         n_burn,
#         start_value=None,
#         seed=None,
#         return_info=False,
#     ):
#         assert isinstance(target, Distribution)
#         assert isinstance(proposal, Distribution)
#         self.target = target
#         self.proposal = proposal

#         self.n_sample = n_sample
#         self.n_burn = n_burn
#         if start_value is None:
#             self.start_value = np.zeros(len(self.target))
#         else:
#             self.start_value = start_value
#         self._rng = np.random.default_rng(seed)
#         self.return_info = return_info

#     def __call__(self):
#         start = 0
#         xi = self.start_value
#         logpdf = self.target.calc_logpdf(xi)
#         g = self.proposal.calc_logpdf(xi)

#         self.samples = np.zeros((self.n_sample + 1, len(self.target)))
#         self.samples[0] = xi

#         if self.return_info:
#             self.logpdfs = np.zeros((self.n_sample + 1))
#             self.logpdfs[0] = logpdf

#         accept_rate = 0.0

#         for i in range(start + 1, self.n_sample + 1):
#             xi_prop = self.proposal.calc_sample(self._rng)
#             logpdf_prop = self.target.calc_logpdf(xi_prop)
#             g_prop = self.proposal.calc_logpdf(xi_prop)
#             logalpha = logpdf_prop - logpdf + g - g_prop

#             if logalpha < 0:
#                 if self._rng.uniform() < np.exp(logalpha):
#                     accept = True
#                 else:
#                     accept = False
#             else:
#                 accept = True

#             if accept:
#                 xi = xi_prop
#                 logpdf = logpdf_prop
#                 g = g_prop
#                 accept_rate += 1 / 100

#             self.samples[i] = xi

#             if self.return_info:
#                 self.logpdfs[i] = logpdf

#             if i % 100 == 0:
#                 print("MCMC sample {} of {}".format(i, self.n_sample))
#                 print(xi)
#                 print(logpdf)
#                 print("Accept rate:", accept_rate)
#                 print("")

#                 accept_rate = 0.0

#         if self.return_info:
#             info = {"loglikelihood": self.logpdfs}
#             return self.samples, info
#         else:
#             return self.samples


target = ArbitraryTarget()
# proposal = GaussianMixture()
proposal = Mixture([Gaussian([4.0], [1.0]), Gaussian([-4.0], [1.0])])

mcmc = IndependenceSampler(
    target=target,
    proposal=proposal,
    n_sample=10000,
    n_burn=0,
    seed=0,
)

samples = mcmc()
counts, bins = np.histogram(samples, bins=100, range=(-10, 10))
counts = counts / np.sum(counts) / (bins[1] - bins[0])

x = np.linspace(-10, 10, 1000)
t = target.calc_pdf(x)
p = proposal.calc_pdf(x)

plt.figure()
plt.plot(x, t)
plt.plot(x, p)
plt.stairs(counts, bins)
plt.show()
