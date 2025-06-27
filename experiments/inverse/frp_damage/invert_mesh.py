import numpy as np
import matplotlib.pyplot as plt
import gmsh

from experiments.inverse.frp_damage import caching, params, misc

n_fiber = params.geometry_params["n_fiber"]
h = 0.100
fibers = caching.get_or_calc_fibers()
print("Computing mesh")
rve_size = params.geometry_params["rve_size"]
r_fiber = params.geometry_params["r_fiber"]

fname1 = "meshes/rve_h-{:.3f}_original.msh".format(h)
misc.create_mesh(fibers=fibers, a=rve_size, r=r_fiber, h=h, fname=fname1, shift=False)

fname2 = "meshes/rve_h-{:.3f}_shifted.msh".format(h)
fibers = caching.get_or_calc_fibers()
misc.create_mesh(fibers=fibers, a=rve_size, r=r_fiber, h=h, fname=fname2, shift=True)

from fem.meshing import read_mesh
from myjivex.util import QuickViewer

nodes1, elems1, egroups1 = read_mesh(fname1, read_groups=True)
nodes2, elems2, egroups2 = read_mesh(fname2, read_groups=True)


def get_edges(elems, egroups):
    fiber_edges = set()
    matrix_edges = set()

    for ielem in egroups["fiber"].get_indices():
        inodes = elems[ielem]
        for i, j in [(0, 1), (1, 2), (2, 0)]:
            if inodes[i] < inodes[j]:
                edge = (inodes[i], inodes[j])
            else:
                edge = (inodes[j], inodes[i])

            fiber_edges.add(edge)

    for ielem in egroups["matrix"].get_indices():
        inodes = elems[ielem]
        for i, j in [(0, 1), (1, 2), (2, 0)]:
            if inodes[i] < inodes[j]:
                edge = (inodes[i], inodes[j])
            else:
                edge = (inodes[j], inodes[i])

            if edge not in fiber_edges:
                matrix_edges.add(edge)

    return fiber_edges, matrix_edges


fiber_edges1, matrix_edges1 = get_edges(elems1, egroups1)
fiber_edges2, matrix_edges2 = get_edges(elems2, egroups2)

fig, ax = plt.subplots()

for i, edges in enumerate([matrix_edges1, fiber_edges1]):
    color = "C" + str(i)

    for edge in edges:
        coords = nodes1[list(edge)]
        ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=0.5)

for i, edges in enumerate([matrix_edges2, fiber_edges2]):
    color = "C" + str(i + 2)

    for edge in edges:
        coords = nodes2[list(edge)]
        ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=0.5)

ax.set_xlim((-0.4, 0.4))
ax.set_ylim((-0.4, 0.4))
ax.set_aspect("equal")

plt.show()
