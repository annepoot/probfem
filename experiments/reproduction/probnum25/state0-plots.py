import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from myjive.fem import XNodeSet, XElementSet
from fem.jive import CJiveRunner
from fem.meshing import calc_boundary_nodes
from rmfem.perturbation import calc_perturbed_coords_cpp

from experiments.reproduction.probnum25.props import get_fem_props, get_rwm_fem_target


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


props = get_fem_props()
globdat_h = run_forward(props, elems_h)
globdat_ref = run_forward(props, elems_ref)

x_h = nodes_h.get_coords().flatten()
x_ref = nodes_ref.get_coords().flatten()

u_h = globdat_h["state0"]
u_ref = globdat_ref["state0"]

fig, ax = plt.subplots()
ax.plot(x_ref, u_ref, color="k", linestyle="-", label=r"$u^*$")

for i in range(100):
    nodes_pert, elems_pert = generate_mesh(n_elem_h)
    pert_coords = calc_perturbed_coords_cpp(
        ref_coords=nodes_h.get_coords(),
        elems=elems_h,
        elem_sizes=np.full(n_elem_h, 1 / n_elem_h),
        p=1,
        boundary=calc_boundary_nodes(elems_h),
        rng=np.random.default_rng(i),
    )
    nodes_pert.set_coords(pert_coords)

    x_pert = nodes_pert.get_coords().flatten()
    u_pert = run_forward(props, elems_pert)["state0"]

    label = r"$\tilde{u}^h$" if i == 0 else None
    ax.plot(x_pert, u_pert, color="C0", linestyle="-", alpha=0.2, label=label)

ax.plot(x_h, u_h, color="0.7", linestyle="-", label=r"$u^h$")

target = get_rwm_fem_target(elems=elems_h, std_corruption=1e-5, sigma_e=1e-5)
u_obs = target.likelihood.values
x_obs = np.linspace(0, 1, len(u_obs) + 2)[1:-1]
ax.scatter(x_obs, u_obs, color="k", zorder=2, label=r"$y$")

ax.set_xlim((0.0, 1.0))
ax.set_ylim((-0.025, 0.022))
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$u$")

patch = Rectangle((0.05, 0.005), 0.4, 0.015, fill=False, linestyle="--", linewidth=1)
ax.add_patch(patch)
plt.text(0.06, 0.018, "Fig.4")
plt.legend()

if len(u_obs) == 4:
    folder = "4-observations"
elif len(u_obs) == 9:
    folder = "9-observations"
else:
    raise ValueError

fname = os.path.join("img", folder, "state0-plot.pdf")
plt.savefig(fname, bbox_inches="tight")
plt.show()
