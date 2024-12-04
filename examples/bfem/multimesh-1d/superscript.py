import numpy as np
import matplotlib.pyplot as plt

from myjive.fem import XElementSet, XNodeSet
from meshing import write_mesh
from fem_props import get_fem_props
from probability.process import (
    GaussianProcess,
    InverseCovarianceOperator,
    ProjectedPrior,
)
from bfem.observation import compute_bfem_observations


def create_random_mesh(rng):
    x = np.linspace(0, 1, 65)
    point = rng.choice(x[1:-1])
    nodes = XNodeSet()
    nodes.add_node([0.0])
    nodes.add_node([point])
    nodes.add_node([1.0])
    nodes.to_nodeset()

    elems = XElementSet(nodes)
    elems.add_element([0, 1])
    elems.add_element([1, 2])
    elems.to_elementset()

    return elems


cprops = get_fem_props("bar_random.mesh")
fprops = get_fem_props("bar_fine.mesh")

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

for i in range(20):
    write_mesh(create_random_mesh(rng), "bar_random.mesh")
    coarse_prior = ProjectedPrior(inf_prior, cprops["init"], cprops["solve"])
    coarse_globdat = coarse_prior.globdat
    u_coarse.append(coarse_globdat["state0"])
    x_coarse.append(coarse_globdat["nodeSet"].get_coords().flatten())

    PhiT = compute_bfem_observations(coarse_prior, fine_prior, fspace=False)
    H_obs = PhiT @ fine_globdat["matrix0"]
    f_obs = PhiT @ fine_globdat["extForce"]

    posterior = posterior.condition_on(H_obs, f_obs)

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
    for i, (uc, xc) in enumerate(zip(u_coarse, x_coarse)):
        label = "coarse solution" if i == 0 else None
        plt.plot(xc, uc, color="C2", label=label, alpha=0.5)
    plt.plot(x, u, color="C3", label="fine solution")
    plt.ylim((-13, 17))
    plt.legend(loc="upper left")
    plt.show()
