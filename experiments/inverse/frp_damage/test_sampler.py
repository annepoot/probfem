import numpy as np
import matplotlib.pyplot as plt

from probability import Likelihood
from probability.univariate import Gaussian as UVGaussian
from probability.multivariate import Gaussian as MVGaussian
from probability.process import GaussianProcess, ZeroMeanFunction, SquaredExponential

from experiments.inverse.frp_damage import misc, params, sampler

domain = np.linspace(0.0, 0.2, 101)

inf_prior = GaussianProcess(
    mean=ZeroMeanFunction(),
    cov=SquaredExponential(l=0.02, sigma=2.0),
)

std_pd = 1e-6
cov = inf_prior.calc_cov(domain, domain) + std_pd**2 * np.identity(len(domain))
prior = MVGaussian(mean=None, cov=cov, use_scipy_latent=False)

rng = np.random.default_rng(0)

E_matrix = params.material_params["E_matrix"]
alpha = params.material_params["alpha"]
beta = params.material_params["beta"]
c = params.material_params["c"]
d = params.material_params["d"]

saturation = misc.saturation(domain, alpha, beta, c)
damage = misc.damage(saturation, d)
E = E_matrix * (1 - damage)

x_obs = 0.05
y_obs = misc.damage(misc.saturation(x_obs, alpha, beta, c), d)

observation = (0.05, 0.4)


class CustomLikelihood(Likelihood):

    def __init__(self):
        self.dist = UVGaussian(y_obs, 1e-2)

    def calc_logpdf(self, x):
        damage = misc.sigmoid(x, 1.0, 0.0)
        idx = int(x_obs / np.max(domain) * (len(domain) - 1))
        pred = damage[idx]
        l = self.dist.calc_logpdf(pred)
        return l


likelihood = CustomLikelihood()

fig, ax = plt.subplots()

damage_samples = []

for i in range(100):
    sample = prior.calc_sample(rng)
    l = likelihood.calc_logpdf(sample)
    damage_sample = misc.sigmoid(sample, 1.0, 0.0)
    damage_samples.append(damage_sample)

    if l < -100:
        alpha = 0.1
    elif l > 100:
        alpha = 1.0
    else:
        alpha = 0.1 + 0.9 * misc.sigmoid(l, 1.0, 0.0)
    ax.plot(domain, damage_sample, color="C0", alpha=alpha)

damage_samples = np.array(damage_samples)

ax.scatter([x_obs], [y_obs], color="k")
ax.plot(domain, damage, color="k")
ax.set_title("Prior samples of damage field")
plt.show()

sampler = sampler.EllipticalSliceSampler(
    prior=prior,
    likelihood=likelihood,
    n_sample=1000,
    n_burn=0,
    return_info=True,
)

samples, info = sampler()

step = 50
n_start = 0
n_end = 1000

fig, ax = plt.subplots()
for i, sample in enumerate(samples[n_start:n_end:step]):
    damage_sample = misc.sigmoid(sample, 1.0, 0.0)
    alpha = i / (n_end - n_start) * step
    ax.plot(domain, damage_sample, color="C0", alpha=alpha)
ax.scatter([x_obs], [y_obs], color="k", zorder=2)
ax.plot(domain, damage, color="k")
ax.set_title("Posterior samples of damage field")
plt.show()
