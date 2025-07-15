import os
import numpy as np
from scipy.spatial import Delaunay, ConvexHull
import ctypes as ct

from myjive.fem import (
    NodeSet,
    XNodeSet,
    ElementSet,
    XElementSet,
    to_xelementset,
    ElementGroup,
)
from fem.meshing import create_bboxes, list_point_bbox_intersections
from fem.jive import libcppbackend

__all__ = [
    "create_convex_triangulation",
    "calc_convex_corner_nodes",
    "calc_convex_boundary_nodes",
    "calc_boundary_nodes",
    "invert_convex_mesh",
    "invert_mesh",
    "check_point_in_shape",
    "check_point_in_interval",
    "check_point_in_polygon",
    "check_point_in_polygon_cpp",
    "get_patch_around_node",
    "get_patches_around_nodes",
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


def calc_boundary_nodes(
    elems, include_corners=True, include_sides=True, tol=1e-8, patches=None, egroup=None
):
    nodes = elems.get_nodes()
    rank = nodes.rank()
    bnodes = {}

    max_node_count = elems.max_elem_node_count()

    if patches is None:
        patches = get_patches_around_nodes(elems, egroup=egroup)

    for inode in range(len(nodes)):
        ipatch = patches[inode]
        patch_nodes = np.zeros((len(ipatch), max_node_count), dtype=int) - 1

        for ie, ielem in enumerate(ipatch):
            patch_nodes[ie] = elems[ielem]

        if rank == 1:
            unique_nodes = np.unique(patch_nodes)

            if patch_nodes.shape[0] == 1:
                if patch_nodes[0, 1] == inode:
                    curr_coords = nodes.get_node_coords(inode)
                    prev_coords = nodes.get_node_coords(patch_nodes[0, 0])

                    dist = curr_coords - prev_coords
                    norm = dist / np.linalg.norm(dist)

                    bnodes[inode] = np.zeros((1, 1), dtype=float)
                    bnodes[inode][0] = norm

                elif patch_nodes[0, 0] == inode:
                    curr_coords = nodes.get_node_coords(inode)
                    next_coords = nodes.get_node_coords(patch_nodes[0, 1])

                    dist = next_coords - curr_coords
                    norm = -dist / np.linalg.norm(dist)

                    bnodes[inode] = np.zeros((1, 1), dtype=float)
                    bnodes[inode][0] = norm

        elif rank == 2:
            unique_nodes = patch_nodes.flatten()
            _, idx, cnt = np.unique(unique_nodes, return_index=True, return_counts=True)
            unique_nodes = unique_nodes[idx[cnt == 1]]

            prev_bnode = next_bnode = -1

            for ie, inodes in enumerate(patch_nodes):
                elem_node_count = len(inodes)

                loc = np.where(inode == inodes)[0]
                assert len(loc) == 1

                prev_inode = inodes[(loc[0] - 1) % elem_node_count]
                next_inode = inodes[(loc[0] + 1) % elem_node_count]

                if prev_inode in unique_nodes:
                    assert prev_bnode == -1
                    prev_bnode = prev_inode

                if next_inode in unique_nodes:
                    assert next_bnode == -1
                    next_bnode = next_inode

            assert (prev_bnode == -1) == (next_bnode == -1)

            if prev_bnode >= 0:
                bnodes[inode] = np.zeros((2, 2), dtype=float)

                prev_coords = nodes[prev_bnode]
                curr_coords = nodes[inode]
                next_coords = nodes[next_bnode]

                dist = prev_coords - curr_coords
                norm = np.array([-dist[1], dist[0]]) / np.linalg.norm(dist)
                bnodes[inode][0] = norm

                dist = curr_coords - next_coords
                norm = np.array([-dist[1], dist[0]]) / np.linalg.norm(dist)
                bnodes[inode][1] = norm

        else:
            assert False

    return bnodes


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


def invert_mesh(mesh):
    if isinstance(mesh, ElementSet):
        elems = mesh
        nodes = elems.get_nodes()
        egroups = {"all": np.arange(len(elems))}
    else:
        assert isinstance(mesh, tuple)
        if len(mesh) == 2:
            nodes, elems = mesh
            egroups = {"all": np.arange(len(elems))}
        elif len(mesh) == 3:
            nodes, elems, egroups = mesh
        else:
            assert False

    assert isinstance(nodes, NodeSet)
    assert isinstance(elems, ElementSet)

    egroups = split_discontiguous_elementgroups((nodes, elems, egroups))
    bboxes = create_bboxes(elems)
    mesh_list = []

    for name, egroup in egroups.items():
        inodesb = calc_boundary_nodes(elems, egroup=egroup)
        mask = np.array([ielem in egroup for ielem in np.arange(len(elems))])

        invnodes = XNodeSet()
        for inodeb in inodesb:
            coords = nodes.get_node_coords(inodeb)
            invnodes.add_node(coords)

        for ielem in egroup:
            inodes = elems[ielem]
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
            ielems = ielems[np.where(mask[ielems])]

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

        invegroup = ElementGroup(invelems)
        invegroup.add_elements(np.arange(len(invelems)))
        invegroups = {name.split("_")[0]: invegroup}

        mesh_list.append((invnodes, invelems, invegroups))

    if len(mesh_list) == 1:
        return mesh_list[1]
    else:
        mesh = merge_meshes(mesh_list)
        return mesh


PTR = ct.POINTER
cpp_func = libcppbackend.check_point_in_polygon
cpp_func.argtypes = (
    PTR(ct.c_double),  # point
    PTR(ct.c_double),  # polygon
    ct.c_int,  # size
    ct.c_double,  # tol
)
cpp_func.restype = ct.c_bool


def check_point_in_polygon_cpp(point, coords, tol=1e-8):
    point_ptr = point.ctypes.data_as(PTR(ct.c_double))
    coords_ptr = coords.ctypes.data_as(PTR(ct.c_double))

    return cpp_func(point_ptr, coords_ptr, len(coords), tol)


def check_point_in_polygon(point, coords, *, tol=1e-8):
    A = coords[-1]
    for B in coords:
        cross = np.cross(B - A, point - A)
        if cross < -tol:
            return False
        A = B
    return True


def check_point_in_interval(point, coords, *, tol=1e-8):
    if np.min(coords) - tol < point[0] < np.max(coords) + tol:
        return True
    else:
        return False


def check_point_in_shape(point, shape, *, tol=1e-8):
    assert len(point) == shape.shape[1]
    if len(point) == 1:
        return check_point_in_interval(point, shape, tol=tol)
    elif len(point) == 2:
        return check_point_in_polygon_cpp(point, shape, tol=tol)
    else:
        raise NotImplementedError


def get_patch_around_node(inode, elems):
    idx0, idx1 = np.where(elems._data == inode)
    return idx0[np.where(elems._groupsizes[idx0] > idx1)]


def get_patches_around_nodes(elems, egroup=None):
    nodes = elems.get_nodes()
    patches = [[] for _ in nodes]

    if egroup is None:
        ielems = np.arange(len(elems))
    else:
        ielems = egroup.get_indices()

    for ielem in ielems:
        inodes = elems[ielem]
        for inode in inodes:
            patches[inode].append(ielem)

    return patches


def split_discontiguous_elementgroups(mesh):
    assert isinstance(mesh, tuple)
    nodes, elems, egroups = mesh

    split_egroups = {}

    for name, egroup in egroups.items():
        nodesubgroups = []
        elemsubgroups = []

        for ielem in egroup.get_indices():
            inodes = elems[ielem]
            matches = set()

            for inode in inodes:
                for igroup, nodesubgroup in enumerate(nodesubgroups):
                    if inode in nodesubgroup:
                        matches.add(igroup)

            matches = sorted(list(matches))
            if len(matches) == 0:
                nodesubgroups.append(set(inodes))
                elemsubgroups.append(set([ielem]))
            elif len(matches) == 1:
                igroup = matches.pop()
                nodesubgroups[igroup].update(inodes)
                elemsubgroups[igroup].add(ielem)
            else:
                imaingroup = matches.pop(0)
                nodesubgroups[imaingroup].update(inodes)
                elemsubgroups[imaingroup].add(ielem)

                # merge subgroups
                for igroup in matches[::-1]:
                    nodesubgroups[imaingroup].update(nodesubgroups.pop(igroup))
                    elemsubgroups[imaingroup].update(elemsubgroups.pop(igroup))

        for i, elemsubgroup in enumerate(elemsubgroups):
            group_name = "{}_{}".format(name, i)
            split_egroups[group_name] = ElementGroup(elems)
            split_egroups[group_name].add_elements(elemsubgroup)

    return split_egroups


def merge_meshes(meshes):
    from fem.meshing.findnode import find_coords_in_nodeset

    joint_nodes = XNodeSet()
    joint_elems = XElementSet(joint_nodes)
    joint_egroups = {}

    for mesh in meshes:
        if len(mesh) == 2:
            nodes, elems = mesh
            do_groups = False
        else:
            assert len(mesh) == 3
            nodes, elems, egroups = mesh
            do_groups = True

        nodemap = {}
        elemmap = {}

        for inode, coords in enumerate(nodes):
            ijointnode = find_coords_in_nodeset(coords, joint_nodes)

            if ijointnode is None:
                ijointnode = joint_nodes.add_node(coords)

            nodemap[inode] = ijointnode

        for ielem, inodes in enumerate(elems):
            ijointnodes = [nodemap[inode] for inode in inodes]
            ijointelem = joint_elems.add_element(ijointnodes)

            elemmap[ielem] = ijointelem

        if do_groups:
            for name, egroup in egroups.items():
                if name in joint_egroups:
                    joint_egroup = joint_egroups[name]
                else:
                    joint_egroup = ElementGroup(joint_elems)
                    joint_egroups[name] = joint_egroup

                for ielem in egroup:
                    ijointelem = elemmap[ielem]
                    joint_egroup.add_element(ijointelem)

    joint_nodes.to_nodeset()
    joint_elems.to_elementset()

    if do_groups:
        return joint_nodes, joint_elems, joint_egroups
    else:
        return joint_nodes, joint_elems
