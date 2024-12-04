import numpy as np
from scipy.spatial import Delaunay, ConvexHull

from myjive.fem import XNodeSet, XElementSet, to_xelementset
from fem.meshing import create_bboxes, list_point_bbox_intersections

__all__ = [
    "create_convex_triangulation",
    "calc_convex_corner_nodes",
    "calc_convex_boundary_nodes",
    "calc_boundary_nodes",
    "invert_convex_mesh",
    "invert_mesh",
    "check_point_in_polygon",
    "get_patch_around_node"
]


def create_convex_triangulation(nodes):
    tri = Delaunay(nodes.get_coords())

    elems = XElementSet(nodes)
    for inodes in tri.simplices:
        elems.add_element(inodes)
    elems.to_elementset()

    return elems


def calc_convex_corner_nodes(nodes):
    hull = ConvexHull(nodes.get_coords())
    return hull.vertices


def calc_convex_boundary_nodes(nodes, include_corners=True, sort=False):
    hull = ConvexHull(nodes.get_coords(), qhull_options="Qc")
    if not sort:
        boundaries = hull.coplanar[:, 0]
        if not include_corners:
            return boundaries
        else:
            corners = hull.vertices
            return np.concatenate([corners, boundaries])
    else:
        corners = hull.vertices
        coplanar = hull.coplanar
        simplices = hull.simplices

        if include_corners:
            boundaries = np.zeros(len(corners) + len(coplanar), dtype=int)
        else:
            boundaries = np.zeros(len(coplanar), dtype=int)

        offset = 0

        for ivertex, A in enumerate(corners):
            B = corners[(ivertex + 1) % len(corners)]

            check1 = np.all(simplices == np.array([A, B]), axis=1)
            check2 = np.all(simplices == np.array([B, A]), axis=1)
            checktot = np.logical_or(check1, check2)

            if np.sum(checktot) != 1:
                raise RuntimeError("no unique simplex founds")

            isimplex = np.where(checktot)[0]
            inodesb = hull.coplanar[:, 0][hull.coplanar[:, 1] == isimplex]
            coordsb = nodes.get_some_coords(inodesb)

            if include_corners:
                boundaries[offset] = A
                offset += 1

            dist = np.sum((coordsb - nodes.get_node_coords(A)) ** 2, axis=1)
            boundaries[offset : offset + len(inodesb)] = inodesb[np.argsort(dist)]
            offset += len(inodesb)

        assert offset == len(boundaries)

        return boundaries


def calc_boundary_nodes(elems, include_corners=True, include_sides=True, tol=1e-8):
    nodes = elems.get_nodes()
    ibnodes = []
    angles = []

    for inode in range(len(nodes)):
        ipatch = get_patch_around_node(inode, elems)
        angle = 0.0

        for ie, ielem in enumerate(ipatch):
            inodes = elems[ielem]
            coords = nodes.get_some_coords(inodes)

            iB = np.where(inodes == inode)[0][0]
            iA = (iB - 1) % len(inodes)
            iC = (iB + 1) % len(inodes)

            BA = coords[iA] - coords[iB]
            BC = coords[iC] - coords[iB]
            theta = np.arctan2(BA[1], BA[0]) - np.arctan2(BC[1], BC[0])

            if theta < 0:
                theta += 2 * np.pi

            if theta < 0 or theta > np.pi:
                raise RuntimeError("concave element angle")

            angle += theta

        if 2 * np.pi - angle > tol:
            if abs(np.pi - angle) < tol:
                include = include_sides
            else:
                include = include_corners

            if include:
                ibnodes.append(inode)

    return np.array(ibnodes, dtype=int)


def invert_convex_mesh(elems):
    nodes = elems.get_nodes()

    inodesb = calc_convex_boundary_nodes(nodes, include_corners=True, sort=True)
    inodesc = calc_convex_corner_nodes(nodes)

    invnodes = XNodeSet()
    for inodec in inodesc:
        coords = nodes.get_node_coords(inodec)
        invnodes.add_node(coords)

    coordsb = nodes.get_some_coords(inodesb)
    coordsb = np.append(coordsb, [coordsb[0]], axis=0)
    midpoints = (coordsb[1:] + coordsb[:-1]) / 2
    for coords in midpoints:
        invnodes.add_node(coords)

    for inodes in elems:
        coords = nodes.get_some_coords(inodes)
        midpoint = np.mean(coords, axis=0)
        invnodes.add_node(midpoint)

    invnodes.to_nodeset()
    invelems = create_convex_triangulation(invnodes)

    return invnodes, invelems


def invert_mesh(elems):
    nodes = elems.get_nodes()
    bboxes = create_bboxes(elems)

    inodesb = calc_boundary_nodes(elems)

    invnodes = XNodeSet()
    for inodeb in inodesb:
        coords = nodes.get_node_coords(inodeb)
        invnodes.add_node(coords)

    for inodes in elems:
        coords = nodes.get_some_coords(inodes)
        midpoint = np.mean(coords, axis=0)
        invnodes.add_node(midpoint)

    invnodes.to_nodeset()
    invelems = create_convex_triangulation(invnodes)

    idelelems = []
    for iinvelem, iinvnodes in enumerate(invelems):
        invcoords = invnodes.get_some_coords(iinvnodes)
        midpoint = np.mean(invcoords, axis=0)

        ielems = list_point_bbox_intersections(midpoint, bboxes)

        if len(ielems) == 0:
            idelelems.append(iinvelem)
        else:
            for ielem in ielems:
                inodes = elems.get_elem_nodes(ielem)
                coords = nodes.get_some_coords(inodes)
                if check_point_in_polygon(midpoint, coords):
                    break
            else:
                idelelems.append(iinvelem)
    idelelems = np.array(idelelems)

    to_xelementset(invelems)
    idelelems = -np.sort(-idelelems)
    for idelelem in idelelems:
        invelems.erase_element(idelelem)
    invelems.to_elementset()

    return invnodes, invelems


def check_point_in_polygon(point, coords, tol=1e-8):
    A = coords[-1]
    for B in coords:
        cross = np.cross(B - A, point - A)
        if cross < -tol:
            return False
        A = B
    return True


def get_patch_around_node(inode, elems):
    idx0, idx1 = np.where(elems._data == inode)
    return idx0[np.where(elems._groupsizes[idx0] > idx1)]
