import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from experiments.reproduction.inverse.pullout_bar.props import get_fem_props
from fem.jive import CJiveRunner
from fem.meshing import (
    mesh_interval_with_line2,
    create_phi_from_globdat,
    calc_elem_sizes,
    calc_boundary_nodes,
)
from rmfem.perturbation import calc_perturbed_coords

props = get_fem_props()


def u_exact(x):
    k = props["model"]["model"]["spring"]["k"]
    E = props["model"]["model"]["elastic"]["material"]["E"]
    f = props["model"]["model"]["neum"]["initLoad"]

    nu = np.sqrt(k / E)
    eps = f / E

    A = eps / (nu * (np.exp(nu) - np.exp(-nu)))

    return A * (np.exp(nu * x) + np.exp(-nu * x))


n_elem = 4
nodes, elems = mesh_interval_with_line2(n=n_elem)
props = get_fem_props()
jive = CJiveRunner(props, elems=elems)

plot_nodes, plot_elems = mesh_interval_with_line2(n=720 * n_elem)
plot_jive_runner = CJiveRunner(props, elems=plot_elems)
plot_globdat = plot_jive_runner()

ref_nodes, ref_elems = mesh_interval_with_line2(n=n_elem)
ref_coords = nodes.get_coords()
elem_sizes = calc_elem_sizes(ref_elems)
boundary = calc_boundary_nodes(ref_elems)

pert_nodes, pert_elems = mesh_interval_with_line2(n=n_elem)

n_elem = 4
n_sample = 1000

rng = np.random.default_rng(0)

us = []
xs = []
u_plots = []
x_plot = plot_nodes.get_coords().flatten()

for _ in range(n_sample):
    pert_coords = calc_perturbed_coords(
        ref_coords=ref_coords,
        elems=elems,
        elem_sizes=elem_sizes,
        p=1,
        boundary=boundary,
        rng=rng,
    )
    pert_nodes._data[:, :] = pert_coords
    jive.update_elems(pert_elems)

    globdat_pert = jive()

    u = globdat_pert["state0"]
    x = globdat_pert["nodeSet"]
    Phi_plot = create_phi_from_globdat(globdat_pert, plot_globdat)
    u_plot = Phi_plot @ u

    us.append(u)
    xs.append(x)
    u_plots.append(u_plot)

us = np.array(us)
xs = np.array(xs)
u_plots = np.array(u_plots)

mean = np.mean(np.array(u_plots), axis=0)
std = np.std(np.array(u_plots), axis=0)

c = sns.color_palette("rocket_r", n_colors=8)[2]
xmarkers = np.linspace(0.0, 1.0, 6)
ymarkers = np.linspace(-1.0, 2.0, 7)

plt.figure()
plt.plot(x_plot, mean, color=c)
plt.plot(x_plot, u_plots[:20].T, color=c, linewidth=0.5)
plt.fill_between(x_plot, mean - 2 * std, mean + 2 * std, color=c, alpha=0.3)
plt.plot(x_plot, u_exact(x_plot), color="k")
plt.xlabel(r"$x$", fontsize=12)
plt.ylabel(r"$u$", fontsize=12)
plt.xticks(xmarkers)
plt.yticks(ymarkers)
plt.ylim(ymarkers[[0, -1]])
fname = os.path.join("img", "rmfem-posterior.pdf")
plt.savefig(fname=fname, bbox_inches="tight")
plt.show()
