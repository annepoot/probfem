from myjivex.util import QuickViewer
from fem_props import get_fem_props
from probability.process import (
    GaussianProcess,
    InverseCovarianceOperator,
    ProjectedPrior,
)
from bfem.nonhierarchical import compute_nonhierarchical_posterior

obs_props = get_fem_props("meshes/plate_r0.msh")
ref_props = get_fem_props("meshes/plate_r1.msh")

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
cov_u_post = posterior.calc_cov()
samples_u_post = posterior.calc_samples(20, 0)

QuickViewer(u_ref, refdat, comp=0, title="Reference solution")
QuickViewer(u_obs, obsdat, comp=0, title="Coarse solution")
QuickViewer(u_post, refdat, comp=0, title="Posterior mean")
QuickViewer(std_u_post, refdat, comp=0, title="Posterior std")

error_est = cov_u_post @ refdat["extForce"]

QuickViewer(error_est, refdat, comp=0, title="Error estimate")
