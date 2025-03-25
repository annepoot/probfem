import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from myjive.fem import XNodeSet, XElementSet
from experiments.reproduction.inverse.pullout_bar.props import get_fem_props
from fem.jive import CJiveRunner


def generate_mesh(n_elem):
    node_coords = np.linspace(0, 1, n_elem + 1).reshape((-1, 1))
    nodes = XNodeSet()
    nodes.add_nodes(node_coords)
    nodes.to_nodeset()

    elem_inodes = np.array([np.arange(0, n_elem), np.arange(1, n_elem + 1)]).T
    elem_sizes = np.full(n_elem, 2)

    elems = XElementSet(nodes)
    elems.add_elements(elem_inodes, elem_sizes)
    elems.to_elementset()

    return nodes, elems


props = get_fem_props()


def u_exact(x):
    k = props["model"]["model"]["spring"]["k"]
    E = props["model"]["model"]["elastic"]["material"]["E"]
    f = props["model"]["model"]["neum"]["initLoad"]

    nu = np.sqrt(k / E)
    eps = f / E

    A = eps / (nu * (np.exp(nu) - np.exp(-nu)))

    return A * (np.exp(nu * x) + np.exp(-nu * x))


n_elems = np.array([1, 2, 4, 8, 16, 32, 64])
us = []

for n_elem in n_elems:
    nodes, elems = generate_mesh(n_elem)
    jive = CJiveRunner(props, elems=elems)
    globdat = jive()
    us.append(globdat["state0"])

u_obs = np.array([u[-1] for u in us])
e_obs = u_exact(1) - u_obs

xmarkers = np.linspace(0.0, 1.0, 6)
ymarkers = np.linspace(-0.25, 1.5, 8)
colors = sns.color_palette("rocket_r", n_colors=8)

plt.rc("text", usetex=True)  # use latex for text
plt.rcParams["text.latex.preamble"] = r"\usepackage{xfrac}"

plt.figure()
for i, (n_elem, u) in enumerate(zip(n_elems, us)):
    x = np.linspace(0, 1, n_elem + 1)
    plt.plot(x, u, label=r"$\sfrac{1}{" + str(n_elem) + "}$", color=colors[i])
plt.plot(x, u_exact(x), color="k")
plt.xlabel(r"$x$", fontsize=12)
plt.ylabel(r"$u$", fontsize=12)
plt.xticks(xmarkers)
plt.yticks(ymarkers)
plt.ylim(ymarkers[[0, -1]])
legend = plt.legend(title=r"$h$")
fontsize = "12"
plt.setp(legend.get_texts(), fontsize=fontsize)
plt.setp(legend.get_title(), fontsize=fontsize)
fname = os.path.join("img", "exact-solution.pdf")
plt.savefig(fname=fname, bbox_inches="tight")
plt.show()
