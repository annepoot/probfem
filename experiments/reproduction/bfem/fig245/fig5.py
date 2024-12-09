import numpy as np

from myjivex.util import QuickViewer
from probability.process import (
    GaussianProcess,
    InverseCovarianceOperator,
    ProjectedPrior,
)
from bfem.observation import compute_bfem_observations
from experiments.reproduction.bfem.fig245.fem_props import get_fem_props

cprops = get_fem_props("meshes/plate_r0.msh")
fprops = get_fem_props("meshes/plate_r1.msh")

inf_cov = InverseCovarianceOperator(model_props=fprops["model"], scale=1.0)
inf_prior = GaussianProcess(None, inf_cov)
fine_prior = ProjectedPrior(
    prior=inf_prior, init_props=fprops["init"], solve_props=fprops["solve"]
)
coarse_prior = ProjectedPrior(
    prior=inf_prior, init_props=cprops["init"], solve_props=cprops["solve"]
)


fglobdat = fine_prior.globdat
f = fglobdat["extForce"]
PhiT = compute_bfem_observations(coarse_prior, fine_prior, fspace=False)
H_obs = PhiT @ fglobdat["matrix0"]
f_obs = PhiT @ f

posterior = fine_prior.condition_on(H_obs, f_obs)
mean_u_post = posterior.calc_mean()
std_u_post = posterior.calc_std()
cov_u_post = posterior.calc_cov()

QuickViewer(
    mean_u_post,
    fglobdat,
    comp=0,
    dpi=600,
    figsize=(7.5, 3),
    title="Posterior mean (Fig. 5a)",
    fname="img/K/mean_state0-x_posterior.png",
)
QuickViewer(
    std_u_post,
    fglobdat,
    comp=0,
    dpi=600,
    figsize=(7.5, 3),
    title="Posterior standard deviation (Fig. 5b)",
    fname="img/K/std_state0-x_posterior.png",
)

pdNoise = 1e-4
cov_u_post += pdNoise**2 * np.identity(len(cov_u_post))

l, Q = np.linalg.eigh(cov_u_post)

newl = l * abs(Q.T @ f)
newcov = Q @ np.diag(newl) @ Q.T
newvar = newcov.diagonal()
newstd = np.sqrt(newvar)

QuickViewer(
    cov_u_post @ f,
    fglobdat,
    comp=0,
    dpi=600,
    figsize=(7.5, 3),
    title="Error recovery (Fig. 5c)",
    fname="img/K/std_state0-x_error_recovered.png",
)
QuickViewer(
    newstd,
    fglobdat,
    comp=0,
    dpi=600,
    figsize=(7.5, 3),
    title="Rescaled posterior standard deviation (Fig. 5d)",
    fname="img/K/std_state0-x_posterior_rescaled.png",
)
