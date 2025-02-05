import numpy as np

from myjivex.util import QuickViewer
from probability.process import (
    GaussianProcess,
    InverseCovarianceOperator,
    ProjectedPrior,
)
from bfem.observation import compute_bfem_observations
from experiments.reproduction.bfem.fig245.fem_props import get_fem_props
from experiments.reproduction.bfem.fig245.cfem_props import get_cfem_props
from fem.jive import CJiveRunner

# cmodule_props = get_fem_props("meshes/plate_r0.msh")
# fmodule_props = get_fem_props("meshes/plate_r1.msh")
# jive_runner = None
# cjive_kws = None
# fjive_kws = None

cmodule_props = get_cfem_props("meshes/plate_r0.msh")
fmodule_props = get_cfem_props("meshes/plate_r1.msh")

cmodel_props = cmodule_props.pop("model")
fmodel_props = fmodule_props.pop("model")

assert cmodel_props == fmodel_props

cjive = CJiveRunner(
    cmodule_props,
    node_count=254,
    elem_count=416,
    rank=2,
    max_elem_node_count=3,
)
fjive = CJiveRunner(
    fmodule_props,
    node_count=924,
    elem_count=1664,
    rank=2,
    max_elem_node_count=3,
)

inf_cov = InverseCovarianceOperator(model_props=fmodel_props, scale=1.0)
inf_prior = GaussianProcess(None, inf_cov)
coarse_prior = ProjectedPrior(prior=inf_prior, jive_runner=cjive)
fine_prior = ProjectedPrior(prior=inf_prior, jive_runner=fjive)

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
