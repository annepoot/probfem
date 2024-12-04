import numpy as np
import matplotlib.pyplot as plt

from myjive.solver import Constrainer
from fem.meshing import create_phi_from_globdat
from fem_props import get_fem_props
from probability.process import (
    GaussianProcess,
    InverseCovarianceOperator,
    ProjectedPrior,
)
from bfem.observation import (
    compute_bfem_observations,
    compute_cg_observations,
    compute_random_observations,
)

cprops = get_fem_props("bar_coarse.mesh")
fprops = get_fem_props("bar_fine.mesh")

inf_prior = GaussianProcess(None, InverseCovarianceOperator(fprops["model"]))
fine_prior = ProjectedPrior(inf_prior, fprops["init"], fprops["solve"])
coarse_prior = ProjectedPrior(inf_prior, cprops["init"], cprops["solve"])

fine_globdat = fine_prior.globdat
K = fine_globdat["matrix0"]
f = fine_globdat["extForce"]
c = fine_globdat["constraints"]
conman = Constrainer(c, K)
Kc = conman.get_output_matrix()
fc = conman.get_rhs(f)

# BFEM observations
PhiT = compute_bfem_observations(coarse_prior, fine_prior, fspace=False)

# # CG observations
# PhiT = compute_cg_observations(K, f, c, renormalize=True, n_obs=64)

# # Random observations
# PhiT = compute_random_observations(65, n_obs=64, seed=0)

# Random observation settings
H_obs = PhiT @ Kc
f_obs = PhiT @ fc

x = np.linspace(0, 1, 65)

# Get the prior and posterior means and standard deviations
u = fine_prior.globdat["state0"]
u_coarse = coarse_prior.globdat["state0"]
Phi = create_phi_from_globdat(coarse_prior.globdat, fine_prior.globdat)
u_coarse = Phi @ u_coarse

u_prior = fine_prior.calc_mean()
std_u_prior = fine_prior.calc_std()
samples_u_prior = fine_prior.calc_samples(n=20, seed=0)

posterior = fine_prior

# Create figure 1 directly using matplotlib
for H_seq, f_seq in zip(H_obs, f_obs):
    posterior = posterior.condition_on(H_seq, f_seq)
    u_post = posterior.calc_mean()
    std_u_post = posterior.calc_std()
    samples_u_post = posterior.calc_samples(n=20, seed=0)
    plt.figure()
    plt.plot(x, u_post, color="C0", label="posterior mean")
    plt.plot(x, u_prior, color="C1", label="prior mean")
    plt.plot(x, samples_u_post.T, color="C0", linewidth=0.2)
    plt.plot(x, samples_u_prior.T, color="C1", linewidth=0.2)
    plt.fill_between(x, u_post - 2 * std_u_post, u_post + 2 * std_u_post, alpha=0.3)
    plt.fill_between(x, u_prior - 2 * std_u_prior, u_prior + 2 * std_u_prior, alpha=0.3)
    plt.plot(x, u_coarse, color="C2", label="coarse solution")
    plt.plot(x, u, color="C3", label="fine solution")
    plt.ylim((-13, 17))
    plt.legend(loc="upper left")
    plt.show()
