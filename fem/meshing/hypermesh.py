import numpy as np
from myjive.fem import XNodeSet, XElementSet, NodeSet, ElementSet, ElementGroup
from .boundingbox import create_bboxes, list_bbox_bbox_intersections


__all__ = ["create_hypermesh", "clip_polygons"]


def create_hypermesh(mesh1, mesh2, do_groups=False):
    if isinstance(mesh1, tuple):
        nodes1 = mesh1[0]
        elems1 = mesh1[1]
        if do_groups:
            egroups1 = mesh1[2]
    else:
        elems1 = mesh1
        nodes1 = elems1.get_nodes()
        assert not do_groups

    if isinstance(mesh2, tuple):
        nodes2 = mesh2[0]
        elems2 = mesh2[1]
    else:
        elems2 = mesh2
        nodes2 = elems2.get_nodes()

    assert isinstance(elems1, ElementSet)
    assert isinstance(nodes1, NodeSet)
    assert isinstance(elems2, ElementSet)
    assert isinstance(nodes2, NodeSet)

    nodes1 = elems1.get_nodes()
    nodes2 = elems2.get_nodes()

    nodesh = XNodeSet()
    nodemap1 = np.zeros(len(nodes1), dtype=int)
    nodemap2 = np.zeros(len(nodes2), dtype=int)

    bboxes1 = create_bboxes(elems1)
    bboxes2 = create_bboxes(elems2)

    coords1 = nodes1.get_coords()
    coords2 = nodes2.get_coords()

    rank = coords1.shape[1]
    if rank != coords2.shape[1]:
        raise RuntimeError("incompatible rank!")

    for inode1, coords in enumerate(nodes1):
        inodeh = nodesh.add_node(coords)
        nodemap1[inode1] = inodeh

    for inode2, coords in enumerate(nodes2):
        distances = np.linalg.norm(coords1 - coords, axis=1)
        inode_closest = np.argmin(distances)
        if not np.isclose(distances[inode_closest], 0.0):
            inodeh = nodesh.add_node(coords)
        else:
            inodeh = nodemap1[inode_closest]
        nodemap2[inode2] = inodeh

    elemsh = XElementSet(nodesh)
    elemmap = []

    for ielem1, (inodes1, lbound1, ubound1) in enumerate(zip(elems1, *bboxes1)):
        coords1 = nodes1.get_some_coords(inodes1)
        ielems2 = list_bbox_bbox_intersections((lbound1, ubound1), bboxes2)

        for ielem2 in ielems2:
            inodes2 = elems2[ielem2]
            coords2 = nodes2.get_some_coords(inodes2)

            # check overlap
            if rank == 1:
                if coords1[0, 0] > coords2[0, 0]:
                    left = coords1[0, 0]
                    ileft = nodemap1[inodes1[0]]
                else:
                    left = coords2[0, 0]
                    ileft = nodemap2[inodes2[0]]

                if coords1[1, 0] < coords2[1, 0]:
                    right = coords1[1, 0]
                    iright = nodemap1[inodes1[1]]
                else:
                    right = coords2[1, 0]
                    iright = nodemap2[inodes2[1]]

                if left < right:
                    elemsh.add_element([ileft, iright])
                    elemmap.append((ielem1, ielem2))

            elif rank == 2:
                intersection = clip_polygons(coords1, coords2)
                nside = len(intersection)

                if nside == 0:
                    continue

                elif nside >= 3:
                    # add one or more elements
                    # triangulation is done as:
                    # (0, 1, 2)
                    # (0, 2, 3)
                    # ...
                    # (0, n-2, n-1)
                    for isubelem in range(nside - 2):
                        indices = [0, isubelem + 1, isubelem + 2]
                        coordsh = nodesh.get_coords()
                        inodesh = np.zeros(3, dtype=int)

                        for i, coord in enumerate(intersection[indices]):
                            distances = np.linalg.norm(coordsh - coord, axis=1)
                            inode_closest = np.argmin(distances)

                            if np.isclose(distances[inode_closest], 0.0):
                                inodeh = inode_closest
                            else:
                                inodeh = nodesh.add_node(coord)

                            inodesh[i] = inodeh

                        elemsh.add_element(inodesh)
                        elemmap.append((ielem1, ielem2))

                else:
                    raise RuntimeError("degenerate polygon")

            else:
                raise NotImplementedError("rank {} is not implemented".format(rank))

    if len(elemsh) != len(elemmap):
        raise RuntimeError("elemmap size mismatch")

    inodes_unique = elemsh.get_unique_nodes_of(np.arange(len(elemsh)))
    inodes_absent = np.delete(np.arange(len(nodesh)), inodes_unique)

    if len(inodes_absent) > 0:
        mapping = {}
        starts = [0] + [i + 1 for i in inodes_absent]
        ends = [i for i in inodes_absent] + [len(nodesh)]

        for shift, (start, end) in enumerate(zip(starts, ends)):
            for inode in range(start, end):
                mapping[inode] = inode - shift

        for inode_absent in inodes_absent:
            nodesh.erase_node(inode_absent)

        for ielem, inodes in enumerate(elemsh):
            for i, inode in enumerate(inodes):
                inodes[i] = mapping[inode]
            elemsh.set_elem_nodes(ielem, inodes)

    if do_groups:
        egroupsh = {}

        for key, egroup in egroups1.items():
            egroupsh[key] = ElementGroup(elemsh)

            for ielemh, (ielem1, ielem2) in enumerate(elemmap):
                if ielem1 in egroup:
                    egroupsh[key].add_element(ielemh)

        meshh = (nodesh, elemsh, egroupsh)
    else:
        meshh = (nodesh, elemsh)

    return meshh, elemmap


def clip_polygons(coords1, coords2, tol=1e-8):
    clip = coords1.copy()

    A = coords2[-1]
    for B in coords2:
        # Check each point in the clipped polygon
        cross = np.cross(B - A, clip - A)
        keep = cross > -tol

        # If not all nodes should be kept, perform
        if np.all(np.logical_not(keep)):
            return np.zeros((0, 2))
        elif not np.all(keep):
            newclip = []
            keepprev = keep[-1]
            coordsprev = clip[-1]

            for keepcurr, coordscurr in zip(keep, clip):
                if keepcurr:
                    if not keepprev:
                        mat = np.column_stack([B - A, -(coordscurr - coordsprev)])
                        vec = coordsprev - A
                        s, t = np.linalg.solve(mat, vec)
                        xcoords = A + s * (B - A)

                        if len(newclip) == 0:
                            newclip.append(xcoords)
                        else:
                            diff = xcoords - newclip[-1]
                            if diff @ diff > tol:
                                newclip.append(xcoords)

                    if len(newclip) == 0:
                        newclip.append(coordscurr)
                    else:
                        diff = coordscurr - newclip[-1]
                        if diff @ diff > tol:
                            newclip.append(coordscurr)

                else:
                    if keepprev:
                        # search for the intersection
                        mat = np.column_stack([B - A, -(coordscurr - coordsprev)])
                        vec = coordsprev - A
                        s, t = np.linalg.solve(mat, vec)
                        xcoords = A + s * (B - A)

                        if len(newclip) == 0:
                            newclip.append(xcoords)
                        else:
                            diff = xcoords - newclip[-1]
                            if diff @ diff > tol:
                                newclip.append(xcoords)

                keepprev = keepcurr
                coordsprev = coordscurr

            if len(newclip) < 3:
                return np.zeros((0, 2))

            diff = newclip[0] - newclip[-1]
            if diff @ diff < tol:
                if len(newclip) <= 3:
                    return np.zeros((0, 2))
                else:
                    newclip.pop(-1)

            clip = np.array(newclip)

        A = B

    return clip
