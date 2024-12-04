from myjivex.util import QuickViewer
from myjive.solver import Constrainer

from fem_props import get_fem_props
from probability.process import (
    GaussianProcess,
    InverseCovarianceOperator,
    ProjectedPrior,
)
from bfem.observation import compute_bfem_observations

cprops = get_fem_props("meshes/plate_r0.msh")
fprops = get_fem_props("meshes/plate_r1.msh")

inf_prior = GaussianProcess(None, InverseCovarianceOperator(fprops["model"]))
fine_prior = ProjectedPrior(inf_prior, fprops["init"], fprops["solve"])
coarse_prior = ProjectedPrior(inf_prior, cprops["init"], cprops["solve"])

fine_globdat = fine_prior.globdat
coarse_globdat = coarse_prior.globdat
u_fine = fine_globdat["state0"]
u_coarse = coarse_globdat["state0"]

K = fine_globdat["matrix0"]
f = fine_globdat["extForce"]
c = fine_globdat["constraints"]
conman = Constrainer(c, K)
Kc = conman.get_output_matrix()
fc = conman.get_rhs(f)

PhiT = compute_bfem_observations(coarse_prior, fine_prior, fspace=False)
H_obs = PhiT @ Kc
f_obs = PhiT @ fc

posterior = fine_prior.condition_on(H_obs, f_obs)

u_prior = fine_prior.calc_mean()
std_u_prior = fine_prior.calc_std()
u_post = posterior.calc_mean()
std_u_post = posterior.calc_std()

QuickViewer(u_coarse, coarse_globdat, comp=0, title="Coarse solution")
QuickViewer(u_fine, fine_globdat, comp=0, title="Coarse solution")
QuickViewer(u_post, fine_globdat, comp=0, title="Posterior mean")
QuickViewer(std_u_post, fine_globdat, comp=0, title="Posterior std")
