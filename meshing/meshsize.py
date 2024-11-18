import numpy as np

__all__ = ["calc_elem_sizes", "calc_elem_size"]


def calc_elem_sizes(elems):
    nodes = elems.get_nodes()
    elem_sizes = np.zeros(len(elems))

    for ielem, inodes in enumerate(elems):
        coords = nodes.get_some_coords(inodes)
        elem_sizes[ielem] = calc_elem_size(coords)

    return elem_sizes


def calc_elem_size(coords):
    # compute the largest distance between nodes
    elem_size = 0.0
    n_coord = len(coords)
    for i in range(n_coord):
        for j in range(i + 1, n_coord):
            icoords = coords[i]
            jcoords = coords[j]
            edge = np.sqrt(np.sum((icoords - jcoords) ** 2))
            if edge > elem_size:
                elem_size = edge

    return elem_size
