import numpy as np

from myjive.fem import XNodeSet, XElementSet
from experiments.inverse.pullout_bar.props import get_fem_props
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

    A = f / (nu * (np.exp(nu) - np.exp(-nu)))

    return A * (np.exp(nu * x) + np.exp(-nu * x))


import matplotlib.pyplot as plt

n_elems = np.array([1, 2, 4, 8, 16, 32, 64, 128])
us = []

for n_elem in n_elems:
    nodes, elems = generate_mesh(n_elem)
    jive = CJiveRunner(props, elems=elems)
    globdat = jive()
    us.append(globdat["state0"])

u_obs = np.array([u[-1] for u in us])
e_obs = u_exact(1) - u_obs

plt.figure()
for n_elem, u in zip(n_elems, us):
    x = np.linspace(0, 1, n_elem + 1)
    plt.plot(x, u, label=r"$N={}$".format(n_elem))
plt.plot(x, u_exact(x), color="k")
plt.legend()
plt.show()

plt.figure()
for n_elem, u in zip(n_elems, us):
    x = np.linspace(0, 1, n_elem + 1)
    e = np.abs(u_exact(x) - u)
    plt.semilogy(x, e, label=r"$N={}$".format(n_elem))
plt.legend()
plt.show()

plt.figure()
plt.loglog(n_elems, e_obs, marker="o")
plt.xlabel(r"$N_{elem}$")
plt.ylabel(r"$u^* - u^h$")
plt.show()
