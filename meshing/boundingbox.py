import numpy as np

__all__ = [
    "create_bbox",
    "create_bboxes",
    "calc_bbox_intersection",
    "calc_bbox_intersections",
]


def create_bbox(coords):
    return np.min(coords, axis=0), np.max(coords, axis=0)


def create_bboxes(elems):
    nodes = elems.get_nodes()
    rank = nodes.rank()

    lbounds = np.zeros((len(elems), rank))
    ubounds = np.zeros((len(elems), rank))

    for ielem, inodes in enumerate(elems):
        coords = nodes.get_some_coords(inodes)
        lbounds[ielem] = np.min(coords, axis=0)
        ubounds[ielem] = np.max(coords, axis=0)

    return lbounds, ubounds


def calc_bbox_intersection(bbox1, bbox2):
    if np.any(bbox1[0] > bbox2[1]):
        return False
    elif np.any(bbox1[1] < bbox2[0]):
        return False
    else:
        return True


def calc_bbox_intersections(bbox1, bboxes2):
    lcheck = np.all(bbox1[0] <= bboxes2[1], axis=1)
    ucheck = np.all(bbox1[1] >= bboxes2[0], axis=1)
    return np.where(np.logical_and(lcheck, ucheck))[0]
