from myjive.solver import Constrainer
from myjivex.util import QuickViewer
from probability.process import (
    GaussianProcess,
    NaturalCovarianceOperator,
    ProjectedPrior,
)
from bfem.observation import compute_bfem_observations
from experiments.reproduction.bfem.fig245.fem_props import get_fem_props
from fem.jive import CJiveRunner
from fem.meshing import read_mesh

_, coarse_elems = read_mesh("meshes/plate_r0.msh")
_, fine_elems = read_mesh("meshes/plate_r1.msh")

module_props = get_fem_props()
model_props = module_props.pop("model")

cjive = CJiveRunner(module_props, elems=coarse_elems)
fjive = CJiveRunner(module_props, elems=fine_elems)

inf_cov = NaturalCovarianceOperator(
    model_props=model_props, scale=1.0, lumped_mass_matrix=False
)
inf_prior = GaussianProcess(None, inf_cov)
coarse_prior = ProjectedPrior(prior=inf_prior, jive_runner=cjive)
fine_prior = ProjectedPrior(prior=inf_prior, jive_runner=fjive)

fglobdat = fine_prior.globdat
K = fglobdat["matrix0"]
f = fglobdat["extForce"]
c = fglobdat["constraints"]
Kc = Constrainer(c, K).get_output_matrix()

H_obs, f_obs = compute_bfem_observations(coarse_prior, fine_prior)
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
