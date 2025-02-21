import numpy as np

__all__ = [
    "create_bbox",
    "create_bboxes",
    "check_bbox_bbox_intersection",
    "check_point_bbox_intersection",
    "list_bbox_bbox_intersections",
    "list_point_bbox_intersections",
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


def check_bbox_bbox_intersection(bbox1, bbox2):
    return np.all(bbox1[0] <= bbox2[1]) and np.all(bbox1[1] >= bbox2[0])


def check_point_bbox_intersection(point, bbox):
    return np.all(point >= bbox[0]) and np.all(point <= bbox[1])


def list_bbox_bbox_intersections(bbox1, bboxes2, tol=0.0):
    lcheck = np.all(bbox1[0] - bboxes2[1] <= tol, axis=1)
    ucheck = np.all(bbox1[1] - bboxes2[0] >= -tol, axis=1)
    return np.where(np.logical_and(lcheck, ucheck))[0]


def list_point_bbox_intersections(point, bboxes, tol=0.0):
    lcheck = np.all(point - bboxes[0] >= -tol, axis=1)
    ucheck = np.all(point - bboxes[1] <= tol, axis=1)
    return np.where(np.logical_and(lcheck, ucheck))[0]
