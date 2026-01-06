import numpy as np

from myjive.fem import XNodeSet, XElementSet

__all__ = [
    "mesh_interval_with_line2",
    "mesh_rectangle_with_quad4",
    "mesh_rectangle_with_tri3",
]


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


def mesh_rectangle_with_tri3(*, n):
    # n = number of elements per edge!
    nring = n + 1
    nodes_per_ring = 4 * np.arange(nring)[::-1]
    nodes_per_ring[-1] = 1
    nodes_cum = np.concatenate([np.array([0]), np.cumsum(nodes_per_ring)])

    nnode_total = 2 * n * (n + 1) + 1
    coords = np.zeros((nnode_total, 2))
    idx = 0

    # node loop
    for iring in range(nring):
        nnode_ring = nodes_per_ring[iring]
        offset = 0.5 * iring / (nring - 1)

        if nnode_ring == 1:
            coords[idx, 0] = 0.5
            coords[idx, 1] = 0.5
            idx += 1

        else:
            assert nnode_ring % 4 == 0
            nnode_qring = nnode_ring // 4

            for quart in range(4):
                if quart == 0:
                    x = np.linspace(
                        0.0 + offset, 1.0 - offset, nnode_qring, endpoint=False
                    )
                    y = np.full(nnode_qring, 0.0 + offset)
                elif quart == 1:
                    x = np.full(nnode_qring, 1.0 - offset)
                    y = np.linspace(
                        0.0 + offset, 1.0 - offset, nnode_qring, endpoint=False
                    )
                elif quart == 2:
                    x = np.linspace(
                        1.0 - offset, 0.0 + offset, nnode_qring, endpoint=False
                    )
                    y = np.full(nnode_qring, 1.0 - offset)
                elif quart == 3:
                    x = np.full(nnode_qring, 0.0 + offset)
                    y = np.linspace(
                        1.0 - offset, 0.0 + offset, nnode_qring, endpoint=False
                    )

                coords[idx : idx + nnode_qring, 0] = x
                coords[idx : idx + nnode_qring, 1] = y

                idx += nnode_qring

    nodes = XNodeSet()
    nodes.add_nodes(coords)
    nodes.to_nodeset()

    nelem_total = 4 * n**2
    inodes = np.zeros((nelem_total, 3), dtype=int)
    idx = 0

    # node loop
    for iring in range(nring - 1):
        if iring > 0:
            nnode_ring = nodes_per_ring[iring]
            this_ring = np.arange(nodes_cum[iring], nodes_cum[iring + 1])
            prev_ring = np.arange(nodes_cum[iring - 1], nodes_cum[iring])

            node1s = np.roll(this_ring, -1)
            node2s = this_ring
            del_idx = np.arange(len(prev_ring))[:: nnode_ring // 4 + 1]
            node3s = np.delete(prev_ring, del_idx)

            nelem_ring = len(this_ring)
            inodes[idx : idx + nelem_ring, 0] = node1s
            inodes[idx : idx + nelem_ring, 1] = node2s
            inodes[idx : idx + nelem_ring, 2] = node3s

            idx += nelem_ring

        if iring < (nring - 1):
            nnode_ring = nodes_per_ring[iring]
            this_ring = np.arange(nodes_cum[iring], nodes_cum[iring + 1])
            next_ring = np.arange(nodes_cum[iring + 1], nodes_cum[iring + 2])

            node1s = this_ring
            node2s = np.roll(this_ring, -1)

            if iring < nring - 2:
                dup_idx = (nring - iring - 2) * np.arange(4)
                node3s = np.sort(np.concatenate([next_ring, next_ring[dup_idx]]))
                node3s = np.roll(node3s, -1)
            else:
                node3s = next_ring[[0, 0, 0, 0]]

            nelem_ring = len(this_ring)
            inodes[idx : idx + nelem_ring, 0] = node1s
            inodes[idx : idx + nelem_ring, 1] = node2s
            inodes[idx : idx + nelem_ring, 2] = node3s

            idx += nelem_ring

    elem_sizes = np.full(nelem_total, 3)

    elems = XElementSet(nodes)
    elems.add_elements(inodes, elem_sizes)
    elems.to_elementset()

    return nodes, elems
