import numpy as np

from myjive.fem import NodeSet
from fem.meshing import (
    create_bboxes,
    list_point_bbox_intersections,
    check_point_in_shape,
)

__all__ = ["find_coords_in_nodeset", "find_coords_in_elementset"]


def find_coords_in_nodeset(coords, nodes, *, tol=1e-8):
    if isinstance(nodes, NodeSet):
        all_coords = nodes.get_coords()
    else:
        all_coords = nodes

    if len(coords.shape) == 1:
        inodes = np.where(np.all(abs(all_coords - coords) < tol, axis=1))[0]
        if len(inodes) == 0:
            return None
        elif len(inodes) == 1:
            return inodes[0]
        else:
            assert False

    elif len(coords.shape) == 2:
        inodes = []
        for coord in coords:
            inodes.append(find_coords_in_nodeset(coord, all_coords, tol=tol))
        return inodes
    else:
        assert False


def find_coords_in_elementset(coords, elems, *, bboxes=None):
    nodes = elems.get_nodes()

    if bboxes is None:
        bboxes = create_bboxes(elems)

    for ielem in list_point_bbox_intersections(coords, bboxes):
        inodes = elems[ielem]

        if check_point_in_shape(coords, nodes.get_some_coords(inodes)):
            return ielem

    assert False
