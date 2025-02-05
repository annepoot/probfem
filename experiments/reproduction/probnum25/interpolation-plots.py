import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from myjive.fem import XNodeSet, XElementSet
from fem.jive import CJiveRunner
from fem.meshing import calc_boundary_nodes
from probability.sampling import MCMCRunner
from rmfem.perturbation import calc_perturbed_coords_cpp

from experiments.reproduction.probnum25.props import get_fem_props, get_rwm_fem_target

n_burn = 10000
n_sample = 20000
std_corruption = 1e-5
n_elem_range = [10, 100]


def generate_mesh(n_elem):
    node_coords = np.linspace(0, 1, n_elem + 1).reshape((-1, 1))
    nodes = XNodeSet()
    nodes.add_nodes(node_coords)

    elem_inodes = np.array([np.arange(0, n_elem), np.arange(1, n_elem + 1)]).T
    elem_sizes = np.full(n_elem, 2)

    elems = XElementSet(nodes)
    elems.add_elements(elem_inodes, elem_sizes)

    return nodes, elems


n_elem_h = 10
n_elem_ref = 1000

nodes_h, elems_h = generate_mesh(n_elem_h)
nodes_ref, elems_ref = generate_mesh(n_elem_ref)

nodes_pert, elems_pert = generate_mesh(n_elem_h)
pert_coords = calc_perturbed_coords_cpp(
    ref_coords=nodes_h.get_coords(),
    elems=elems_h,
    elem_sizes=np.full(n_elem_h, 1 / n_elem_h),
    p=1,
    boundary=calc_boundary_nodes(elems_h),
    rng=np.random.default_rng(0),
)
nodes_pert.set_coords(pert_coords)


def run_forward(props, elems):
    jive = CJiveRunner(props, elems=elems)
    globdat = jive()
    return globdat


def run_inverse(elems):
    target = get_rwm_fem_target(elems=elems, std_corruption=1e-5, sigma_e=1e-5)
    proposal = deepcopy(target.prior)

    mcmc = MCMCRunner(
        target=target,
        proposal=proposal,
        n_sample=n_sample,
        n_burn=n_burn,
        start_value=None,
        seed=0,
        recompute_logpdf=False,
    )
    samples = mcmc()

    return samples


props = get_fem_props()
globdat_h = run_forward(props, elems_h)
globdat_ref = run_forward(props, elems_ref)
globdat_pert = run_forward(props, elems_pert)

samples_h = run_inverse(elems_h)
samples_pert = run_inverse(elems_pert)

x_h = nodes_h.get_coords().flatten()
x_ref = nodes_ref.get_coords().flatten()
x_pert = nodes_pert.get_coords().flatten()

u_h = globdat_h["state0"]
u_ref = globdat_ref["state0"]
u_pert = globdat_pert["state0"]

plt.rc("text", usetex=True)
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{xfrac} \usepackage{amsfonts} \usepackage{bm}"
)

fig, ax = plt.subplots()
ax.plot(x_ref, u_ref, color="k", linestyle="-", label=r"$u^*$")
ax.plot(x_h, u_h, color="0.5", linestyle="-", label=r"$u^h$")
ax.plot(x_pert, u_pert, color="C0", linestyle="-", label=r"$\tilde{u}^h$")

sample_h_MAP = np.mean(samples_h[n_burn:], axis=0)
props["model"]["model"]["elastic"]["material"]["params"]["values"][:4] = sample_h_MAP
state0_sample = run_forward(props, elems_h)["state0"]
ax.plot(
    x_h,
    state0_sample,
    color="0.5",
    linestyle="--",
    label=r"$\mathbb{E}[p(u^h|\boldsymbol{y})]$",
)

sample_pert_MAP = np.mean(samples_pert[n_burn:], axis=0)
props["model"]["model"]["elastic"]["material"]["params"]["values"][:4] = sample_pert_MAP
state0_sample = run_forward(props, elems_pert)["state0"]
ax.plot(
    x_pert,
    state0_sample,
    color="C0",
    linestyle="--",
    label=r"$\mathbb{E}[p(\tilde{u}^h|\bm{y})]$",
)

target = get_rwm_fem_target(elems=elems_h, std_corruption=1e-5, sigma_e=1e-5)
u_obs = target.likelihood.values
x_obs = np.linspace(0, 1, len(u_obs) + 2)[1:-1]
ax.scatter(x_obs, u_obs, color="k", zorder=2)

ax.set_xlim((0.05, 0.45))
ax.set_ylim((0.005, 0.020))
ax.set_xticks([0.05, 0.15, 0.25, 0.35, 0.45])
ax.set_yticks([0.005, 0.010, 0.015, 0.020])
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$u$")

plt.legend()

if len(u_obs) == 4:
    folder = "4-observations"
elif len(u_obs) == 9:
    folder = "9-observations"
else:
    raise ValueError

fname = os.path.join("img", folder, "interpolation-plot.pdf")
plt.savefig(fname, bbox_inches="tight")
plt.show()
