from myjivex.util import QuickViewer
from probability.process import (
    GaussianProcess,
    NaturalCovarianceOperator,
    ProjectedPrior,
)
from bfem.observation import compute_bfem_observations
from fem_props import get_fem_props

cprops = get_fem_props("meshes/plate_r0.msh")
fprops = get_fem_props("meshes/plate_r1.msh")

inf_cov = NaturalCovarianceOperator(fprops["model"], lumped_mass_matrix=False)
inf_prior = GaussianProcess(None, inf_cov)
fine_prior = ProjectedPrior(inf_prior, fprops["init"], fprops["solve"])
coarse_prior = ProjectedPrior(inf_prior, cprops["init"], cprops["solve"])

fglobdat = fine_prior.globdat
PhiT = compute_bfem_observations(coarse_prior, fine_prior, fspace=False)
H_obs = PhiT @ fglobdat["matrix0"]
f_obs = PhiT @ fglobdat["extForce"]

posterior = fine_prior.condition_on(H_obs, f_obs)
mean_u_post = posterior.calc_mean()
std_u_post = posterior.calc_std()

QuickViewer(
    mean_u_post,
    fglobdat,
    comp=0,
    dpi=600,
    figsize=(7.5, 3),
    title="Posterior mean (Fig. 4a)",
    fname="img/M/mean_state0-x_posterior.png",
)
QuickViewer(
    std_u_post,
    fglobdat,
    comp=0,
    dpi=600,
    figsize=(7.5, 3),
    title="Posterior standard deviation (Fig. 4b)",
    fname="img/M/std_state0-x_posterior.png",
)
