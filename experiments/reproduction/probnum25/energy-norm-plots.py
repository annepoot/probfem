import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from myjive.fem import XNodeSet, XElementSet
from fem.jive import CJiveRunner
from fem.meshing import calc_boundary_nodes
from rmfem.perturbation import calc_perturbed_coords_cpp

from experiments.reproduction.probnum25.props import get_fem_props


def generate_mesh(n_elem):
    node_coords = np.linspace(0, 1, n_elem + 1).reshape((-1, 1))
    nodes = XNodeSet()
    nodes.add_nodes(node_coords)

    elem_inodes = np.array([np.arange(0, n_elem), np.arange(1, n_elem + 1)]).T
    elem_sizes = np.full(n_elem, 2)

    elems = XElementSet(nodes)
    elems.add_elements(elem_inodes, elem_sizes)

    return nodes, elems


def run_forward(props, elems):
    jive = CJiveRunner(props, elems=elems)
    globdat = jive()
    return globdat


n_sample = 1000
n_elem_range = [10, 20, 40]
E_h_range = []
n_elem_ref = 10000

props = get_fem_props()

dfs = []

for n_elem in n_elem_range:
    E_norms = np.zeros(n_sample)

    nodes_h, elems_h = generate_mesh(n_elem)
    globdat_h = run_forward(props, elems_h)
    u_h = globdat_h["state0"]
    f_h = globdat_h["extForce"]
    E_h_range.append(u_h @ f_h)

    boundary_h = calc_boundary_nodes(elems_h)

    for i in range(n_sample):
        nodes_pert, elems_pert = generate_mesh(n_elem)
        pert_coords = calc_perturbed_coords_cpp(
            ref_coords=nodes_h.get_coords(),
            elems=elems_h,
            elem_sizes=np.full(n_elem, 1 / n_elem),
            p=1,
            boundary=boundary_h,
            rng=np.random.default_rng(i),
        )
        nodes_pert.set_coords(pert_coords)

        globdat_pert = run_forward(props, elems_pert)
        u_pert = globdat_pert["state0"]
        f_pert = globdat_pert["extForce"]

        E_norms[i] = u_pert @ f_pert

    df = pd.DataFrame({"E": E_norms})
    df["n_elem"] = str(n_elem)
    dfs.append(df)

df = pd.concat(dfs)

nodes_ref, elems_ref = generate_mesh(n_elem_ref)
globdat_ref = run_forward(props, elems_ref)
u_ref = globdat_ref["state0"]
f_ref = globdat_ref["extForce"]
E_ref = u_ref @ f_ref
y_coord = 800

plt.rc("text", usetex=True)  # use latex for text
plt.rcParams["text.latex.preamble"] = r"\usepackage{xfrac}"

fig, ax = plt.subplots()
sns.kdeplot(data=df, x="E", hue="n_elem", ax=ax)
ax.scatter(E_ref, y_coord, color="k", marker="x")
for i, E_h in enumerate(E_h_range):
    ax.scatter(E_h, y_coord, color="C" + str(i), marker="x")
ax.get_yaxis().set_visible(False)
ax.set_xlabel(r"$E$")
ax.set_xlim((0.0090, 0.0098))
ax.set_xticks([0.0090, 0.0092, 0.0094, 0.0096, 0.0098])
labels = [r"$\sfrac{1}{10}$", r"$\sfrac{1}{20}$", r"$\sfrac{1}{40}$"]
ax.legend(ax.get_lines()[::-1], labels)
ax.get_legend().set_title(r"$h$")
plt.savefig("img/energy-norm-plot.pdf", bbox_inches="tight")
plt.show()
