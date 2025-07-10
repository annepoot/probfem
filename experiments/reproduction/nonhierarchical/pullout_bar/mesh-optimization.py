import numpy as np
import matplotlib.pyplot as plt

from myjive.fem import XNodeSet, XElementSet

from fem.meshing import mesh_interval_with_line2

from experiments.reproduction.nonhierarchical.pullout_bar import misc

n_elem = 4
obs_nodes, obs_elems = mesh_interval_with_line2(n=n_elem)

n_grid = 256
n_step = 10
opt_coords = np.array([[0.0], [1.0]])
norms = np.zeros(n_grid + 1)
norms_list = np.zeros((n_step, n_grid + 1))

opt_nodes = XNodeSet()
for coord in opt_coords:
    opt_nodes.add_node(coord)

opt_elems = XElementSet(opt_nodes)
opt_elems.add_element([0, 1])

n_ref = len(opt_elems) + 1
ref_coords = np.zeros((n_ref + 1, 1))
ref_coords[:n_ref] = opt_coords

available_nodes = np.arange(1, n_grid)


for i in range(n_step):
    norms[:] = np.nanmax(norms)

    for inode in available_nodes:
        new_coord = np.array([inode / n_grid])
        ref_coords[n_ref] = new_coord

        ref_nodes = XNodeSet()
        for coord in ref_coords:
            ref_nodes.add_node(coord)
        ref_nodes.to_nodeset()

        inodes = np.argsort(ref_coords, axis=0)
        ref_elems = XElementSet(ref_nodes)
        for j in range(n_ref):
            ref_elems.add_element(inodes[j : j + 2].flatten())
        ref_elems.to_elementset()

        norm = misc.calc_norm(obs_elems, ref_elems)
        norms[inode] = norm

    opt_inodes = np.where(np.isclose(np.nanmax(norms), norms))[0]
    opt_coords = np.linspace(0, 1, n_grid + 1)[opt_inodes]

    for opt_coord in opt_coords:
        opt_nodes.add_node(opt_coord)

    idx = np.any(np.subtract.outer(opt_inodes, available_nodes) == 0, axis=0)
    available_nodes = np.delete(available_nodes, np.where(idx))

    n_new = len(opt_inodes)
    ref_coords = np.resize(ref_coords, (n_ref + n_new + 1, 1))
    ref_coords[n_ref : n_ref + n_new, 0] = opt_coords
    n_ref += len(opt_inodes)

    norms_list[i] = norms


opt_coords = opt_nodes.get_coords()

fig, ax = plt.subplots()

for i, norms in enumerate(norms_list):
    ax.plot(norms)
    opt_inodes = (n_grid * opt_nodes[2 * (i + 1) : 2 * (i + 2)].flatten()).astype(int)
    ax.scatter(opt_inodes, norms[opt_inodes], color="k")
ax.axvline(0.00 * n_grid, color="0.5")
ax.axvline(0.25 * n_grid, color="0.5")
ax.axvline(0.50 * n_grid, color="0.5")
ax.axvline(0.75 * n_grid, color="0.5")
ax.axvline(1.00 * n_grid, color="0.5")
plt.show()
