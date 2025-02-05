import numpy as np

from myjive.fem import XNodeSet, XElementSet

__all__ = ["mesh_interval_with_line2", "mesh_rectangle_with_quad4"]

def mesh_interval_with_line2(*, n, L=1.0):
    node_coords = np.linspace(0, L, n + 1).reshape((-1, 1))
    nodes = XNodeSet()
    nodes.add_nodes(node_coords)
    nodes.to_nodeset()

    elem_inodes = np.array([np.arange(0, n), np.arange(1, n + 1)]).T
    elem_sizes = np.full(n, 2)

    elems = XElementSet(nodes)
    elems.add_elements(elem_inodes, elem_sizes)
    elems.to_elementset()

    return nodes, elems


def mesh_rectangle_with_quad4(*, nx, ny, Lx=1.0, Ly=1.0):
    node_coords_x = np.tile(np.linspace(0, Lx, nx + 1), ny + 1)
    node_coords_y = np.repeat(np.linspace(0, Ly, ny + 1), nx + 1)
    node_coords = np.array([node_coords_x, node_coords_y]).T
    nodes = XNodeSet()
    nodes.add_nodes(node_coords)
    nodes.to_nodeset()

    inodes_0 = np.arange(0, (nx + 1) * (ny))
    inodes_0 = inodes_0.reshape((ny, -1))[:, :-1].flatten()
    inodes_1 = inodes_0 + 1
    inodes_2 = inodes_0 + nx + 2
    inodes_3 = inodes_0 + nx + 1
    elem_inodes = np.array([inodes_0, inodes_1, inodes_2, inodes_3]).T
    elem_sizes = np.full(nx * ny, 4)

    elems = XElementSet(nodes)
    elems.add_elements(elem_inodes, elem_sizes)
    elems.to_elementset()

    return nodes, elems

