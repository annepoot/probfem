import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from myjive.fem import NodeSet, ElementSet

from experiments.inverse.frp_damage import caching, params

n_fiber = params.geometry_params["n_fiber"]
h = 0.100
fibers = caching.get_or_calc_fibers()
print("Computing mesh")
rve_size = params.geometry_params["rve_size"]
r_fiber = params.geometry_params["r_fiber"]

mesh = caching.get_or_calc_mesh(h=h)
meshr1 = caching.get_or_calc_mesh(h="{:.3f}r1".format(h))
meshd1 = caching.get_or_calc_dual_mesh(h="{:.3f}d1".format(h))
meshd2 = caching.get_or_calc_dual_mesh(h="{:.3f}d2".format(h))
meshh1 = caching.get_or_calc_hyper_mesh(h="{:.3f}h1".format(h), do_groups=True)
meshh2 = caching.get_or_calc_hyper_mesh(h="{:.3f}h2".format(h), do_groups=True)


def get_edges(mesh):
    if isinstance(mesh, ElementSet):
        do_groups = False
        elems = mesh
        nodes = elems.get_nodes()
    elif isinstance(mesh, tuple):
        do_groups = True
        nodes, elems, egroups = mesh

    assert isinstance(nodes, NodeSet)
    assert isinstance(elems, ElementSet)

    if do_groups:
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

    else:
        edges = set()

        for ielem, inodes in enumerate(elems):
            for i, j in [(0, 1), (1, 2), (2, 0)]:
                if inodes[i] < inodes[j]:
                    edge = (inodes[i], inodes[j])
                else:
                    edge = (inodes[j], inodes[i])

                edges.add(edge)

        return edges


def plot_mesh(mesh1, mesh2=None, *, fname=None):
    if isinstance(mesh1, ElementSet):
        elems1 = mesh1
        nodes1 = elems1.get_nodes()
    else:
        nodes1, elems1, egroups1 = mesh1

    edges1 = get_edges(elems1)

    if mesh2 is not None:
        if isinstance(mesh2, ElementSet):
            elems2 = mesh2
            nodes2 = elems2.get_nodes()
        else:
            nodes2, elems2, egroups2 = mesh2

        edges2 = get_edges(elems2)

    fig, ax = plt.subplots()

    for ielem in egroups1["fiber"].get_indices():
        inodes = elems1[ielem]
        coords = nodes1[inodes]
        patch = Polygon(coords, color="0.8")
        ax.add_patch(patch)

    if mesh2 is None:
        for edge in edges1:
            coords = nodes1[list(edge)]
            ax.plot(coords[:, 0], coords[:, 1], color="k", linewidth=0.5)
    else:
        for edge in edges1:
            coords = nodes1[list(edge)]
            ax.plot(coords[:, 0], coords[:, 1], color="0.4", linewidth=0.5)

        for edge in edges2:
            coords = nodes2[list(edge)]
            ax.plot(coords[:, 0], coords[:, 1], color="k", linewidth=0.5)

    ax.set_xlim((-0.4, 0.4))
    ax.set_ylim((-0.4, 0.4))
    ax.set_aspect("equal")
    ax.set_axis_off()

    if fname is not None:
        plt.savefig(fname=fname, bbox_inches="tight")

    plt.show()


fname = os.path.join("img", "rve-mesh_h-{:.3f}.pdf".format(h))
plot_mesh(mesh, fname=fname)

fname = os.path.join("img", "rve-mesh_h-{:.3f}r1.pdf".format(h))
plot_mesh(meshr1, fname=fname)

fname = os.path.join("img", "rve-mesh_h-{:.3f}d1.pdf".format(h))
plot_mesh(mesh, meshd1, fname=fname)

fname = os.path.join("img", "rve-mesh_h-{:.3f}h1.pdf".format(h))
plot_mesh(meshh1, fname=fname)

fname = os.path.join("img", "rve-mesh_h-{:.3f}d2.pdf".format(h))
plot_mesh(mesh, meshd2, fname=fname)

fname = os.path.join("img", "rve-mesh_h-{:.3f}h2.pdf".format(h))
plot_mesh(meshh2, fname=fname)
