import matplotlib.pyplot as plt

from fem_props import get_fem_props
from probability.process import (
    GaussianProcess,
    InverseCovarianceOperator,
    ProjectedPrior,
)
from bfem.nonhierarchical import compute_nonhierarchical_posterior

obs_props = get_fem_props("bar_coarse.mesh")
ref_props = get_fem_props("bar_ref.mesh")

inf_prior = GaussianProcess(None, InverseCovarianceOperator(ref_props["model"]))
ref_prior = ProjectedPrior(inf_prior, ref_props["init"], ref_props["solve"])
obs_prior = ProjectedPrior(inf_prior, obs_props["init"], obs_props["solve"])
posterior = compute_nonhierarchical_posterior(obs_prior, ref_prior)

refdat = ref_prior.globdat
obsdat = obs_prior.globdat

x_ref = refdat["nodeSet"].get_coords().flatten()
x_obs = obsdat["nodeSet"].get_coords().flatten()
u_ref = refdat["state0"]
u_obs = obsdat["state0"]

u_prior = ref_prior.calc_mean()
std_u_prior = ref_prior.calc_std()
samples_u_prior = ref_prior.calc_samples(20, 0)
u_post = posterior.calc_mean()
std_u_post = posterior.calc_std()
samples_u_post = posterior.calc_samples(20, 0)

# Create figure 1 directly using matplotlib
plt.figure()
plt.plot(x_ref, u_post, color="C0", label="posterior mean")
plt.plot(x_ref, u_prior, color="C1", label="prior mean")
plt.plot(x_ref, samples_u_post.T, color="C0", linewidth=0.2)
plt.plot(x_ref, samples_u_prior.T, color="C1", linewidth=0.2)
plt.fill_between(x_ref, u_post - 2 * std_u_post, u_post + 2 * std_u_post, alpha=0.3)
plt.fill_between(x_ref, u_prior - 2 * std_u_prior, u_prior + 2 * std_u_prior, alpha=0.3)
plt.plot(x_obs, u_obs, color="C2", label="coarse solution")
plt.plot(x_ref, u_ref, color="C3", label="fine solution")
plt.ylim((-13, 17))
plt.legend(loc="upper left")
plt.show()
