import numpy as np

from myjive.fem import XElementSet, XNodeSet
from myjivex.util import QuickViewer
from meshing import write_mesh, create_phi_from_globdat
from fem_props import get_fem_props
from probability.process import (
    GaussianProcess,
    InverseCovarianceOperator,
    ProjectedPrior,
)
from bfem.observation import compute_bfem_observations


def create_random_mesh(rng):
    nodes = XNodeSet()
    nodes.add_node([0.0, 0.0])
    nodes.add_node([1.0, 0.0])
    nodes.add_node([1.0, 1.0])
    nodes.add_node([0.0, 1.0])
    nodes.add_node([0.5, 0.0])
    nodes.add_node([1.0, 0.5])
    nodes.add_node([0.5, 1.0])
    nodes.add_node([0.0, 0.5])
    nodes.add_node(0.25 + 0.5 * rng.random(2))
    nodes.to_nodeset()

    elems = XElementSet(nodes)
    elems.add_element([0, 4, 8, 7])
    elems.add_element([7, 8, 6, 3])
    elems.add_element([4, 1, 5, 8])
    elems.add_element([8, 5, 2, 6])
    elems.to_elementset()

    return elems


cprops = get_fem_props("meshes/box_random.msh")
fprops = get_fem_props("meshes/box_r3.msh")

inf_prior = GaussianProcess(None, InverseCovarianceOperator(fprops["model"]))
fine_prior = ProjectedPrior(inf_prior, fprops["init"], fprops["solve"])
fine_globdat = fine_prior.globdat
u = fine_globdat["state0"]

u_prior = fine_prior.calc_mean()
std_u_prior = fine_prior.calc_std()
samples_u_prior = fine_prior.calc_samples(n=20, seed=0)

posterior = fine_prior
rng = np.random.default_rng(0)
x = np.linspace(0, 1, 65)

u_coarse = []
x_coarse = []

for i in range(10):
    write_mesh(create_random_mesh(rng), "meshes/box_random.msh")
    coarse_prior = ProjectedPrior(inf_prior, cprops["init"], cprops["solve"])
    coarse_globdat = coarse_prior.globdat
    u_coarse.append(coarse_globdat["state0"])
    x_coarse.append(coarse_globdat["nodeSet"].get_coords().flatten())

    PhiT = compute_bfem_observations(coarse_prior, fine_prior, fspace=False)
    H_obs = PhiT @ fine_globdat["matrix0"]
    f_obs = PhiT @ fine_globdat["extForce"]

    posterior = posterior.condition_on(H_obs, f_obs)

    Phi = create_phi_from_globdat(coarse_globdat, fine_globdat)
    uc = Phi @ coarse_globdat["state0"]
    QuickViewer(uc, fine_globdat, comp=0, title=f"Coarse solution {i}", linewidth=0.2)

u_post = posterior.calc_mean()
std_u_post = posterior.calc_std()

QuickViewer(u, fine_globdat, comp=0, title="Fine solution", linewidth=0.2)
QuickViewer(u_post, fine_globdat, comp=0, title="Posterior mean", linewidth=0.2)
QuickViewer(std_u_post, fine_globdat, comp=0, title="Posterior std", linewidth=0.2)
