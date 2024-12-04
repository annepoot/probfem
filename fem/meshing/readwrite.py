import os
import numpy as np

from myjive.fem import XNodeSet, XElementSet

__all__ = ["write_mesh", "read_mesh"]


def write_mesh(elems, fname):
    file, extension = os.path.splitext(fname)

    if extension == ".mesh":
        _write_manual(elems, fname)
    elif extension == ".msh":
        _write_gmsh(elems, fname)
    else:
        raise ValueError("Invalid file type passed to write_mesh")


def read_mesh(fname):
    file, extension = os.path.splitext(fname)

    if extension == ".mesh":
        return _read_manual(fname)
    elif extension == ".msh":
        return _read_gmsh(fname)
    else:
        raise ValueError("Invalid file type passed to write_mesh")


def _write_manual(elems, fname):
    nodes = elems.get_nodes()

    path, file = os.path.split(fname)
    if len(path) > 0 and not os.path.isdir(path):
        os.makedirs(path)

    with open(fname, "w") as file:
        file.write("nodes (ID, x, [y], [z])\n")
        for inode, coords in enumerate(nodes):
            node_id = nodes.get_node_id(inode)
            file.write("{} ".format(node_id))
            file.write(" ".join(["{}".format(coord) for coord in coords]))
            file.write("\n")

        file.write("elements (node#1, node#2, [node#3, ...])\n")
        for ielem, inodes in enumerate(elems):
            node_ids = nodes.get_node_ids(inodes)
            file.write(" ".join(["{}".format(node_id) for node_id in node_ids]))
            file.write("\n")


def _write_gmsh(elems, fname):
    nodes = elems.get_nodes()
    rank = nodes.rank()
    rank3coords = np.zeros(3)

    path, file = os.path.split(fname)
    if len(path) > 0 and not os.path.isdir(path):
        os.makedirs(path)

    with open(fname, "w") as file:
        file.write("$MeshFormat\n")
        file.write("2.2 0 8\n")
        file.write("$EndMeshFormat\n")

        file.write("$Nodes\n")
        file.write("{}\n".format(len(nodes)))
        for inode, coords in enumerate(nodes):
            node_id = nodes.get_node_id(inode)
            rank3coords[:rank] = coords
            file.write("{} ".format(node_id))
            file.write(" ".join(str(coord) for coord in rank3coords))
            file.write("\n")
        file.write("$EndNodes\n")

        file.write("$Elements\n")
        file.write("{}\n".format(len(elems)))
        for ielem, inodes in enumerate(elems):
            node_ids = nodes.get_node_ids(inodes)
            elem_id = elems.get_elem_id(ielem)
            elem_type = get_gmsh_elem_type(rank, len(inodes))
            file.write("{} {} 2 1 1 ".format(elem_id, elem_type))
            file.write(" ".join(str(node_id) for node_id in node_ids))
            file.write("\n")
        file.write("$EndElements\n")


def _read_gmsh(fname):
    nodes = XNodeSet()
    elems = XElementSet(nodes)

    parse_nodes = False
    parse_elems = False
    ranks = []

    with open(fname) as msh:
        for line in msh:
            sp = line.split()

            if line == "$Nodes\n":
                parse_nodes = True
            elif line == "$EndNodes\n":
                parse_nodes = False
            elif line == "$Elements\n":
                parse_elems = True
            elif line == "$EndElements\n":
                parse_elems = False

            if parse_nodes and len(sp) > 1:
                node_id = int(sp[0])
                coords = np.array(sp[1:4], dtype=float)
                nodes.add_node(coords, node_id)

            if parse_elems and len(sp) > 1:
                elem_id = int(sp[0])
                elem_type = int(sp[1])
                elem_rank, nodecount = get_gmsh_elem_info(elem_type)
                ranks.append(elem_rank)
                elem_tagcount = int(sp[2])

                inodes = np.array(sp[3 + elem_tagcount :], dtype=int)
                if len(inodes) != nodecount:
                    raise ValueError("nodecount mismatch")

                elems.add_element(nodes.find_nodes(inodes), elem_id)

        mesh_rank = np.max(ranks)
        coords = nodes.get_coords()
        if coords.shape[1] > mesh_rank:
            nodes.set_coords(coords[:, :mesh_rank])

    nodes.to_nodeset()
    elems.to_elementset()

    return nodes, elems


def _read_manual(fname):
    nodes = XNodeSet()
    elems = XElementSet(nodes)

    parse_nodes = False
    parse_elems = False

    with open(fname) as msh:
        for line in msh:
            sp = line.split()

            if "nodes" in line:
                parse_nodes = True
                parse_elems = False

            elif "elements" in line or "elems" in line:
                parse_nodes = False
                parse_elems = True

            elif parse_nodes and len(sp) > 1:
                nodes.add_node(sp[1:], sp[0])

            elif parse_elems and len(sp) > 0:
                inodes = nodes.find_nodes(sp)
                elems.add_element(inodes)

    # Convert the XNodeSet and XElementSet to a normal NodeSet and ElementSet
    nodes.to_nodeset()
    elems.to_elementset()

    return nodes, elems


gmsh_elem_list = [
    None,
    (1, 2),  # 1:  2-node line.
    (2, 3),  # 2:  3-node triangle.
    (2, 4),  # 3:  4-node quadrangle.
    (3, 4),  # 4:  4-node tetrahedron.
    (3, 8),  # 5:  8-node hexahedron.
    (3, 6),  # 6:  6-node prism.
    (3, 5),  # 7:  5-node pyramid.
    (1, 3),  # 8:  3-node second order line.
    (2, 6),  # 9:  6-node second order triangle.
    (2, 9),  # 10: 9-node second order quadrangle.
    (3, 10),  # 11: 10-node second order tetrahedron.
    (3, 27),  # 12: 27-node second order hexahedron.
    (3, 18),  # 13: 18-node second order prism.
    (3, 14),  # 14: 14-node second order pyramid.
    (0, 1),  # 15: 1-node point.
    (2, 8),  # 16: 8-node second order quadrangle.
    (3, 20),  # 17: 20-node second order hexahedron.
    (3, 15),  # 18: 15-node second order prism.
    (3, 13),  # 19: 13-node second order pyramid.
]


def get_gmsh_elem_info(elem_type):
    if elem_type < 1 or elem_type > 19:
        raise ValueError("invalide elem_type")
    rank, nodecount = gmsh_elem_list[elem_type]
    return rank, nodecount


def get_gmsh_elem_type(rank, node_count):
    if (rank, node_count) not in gmsh_elem_list:
        raise ValueError("invalide rank-nodecount combination")
    else:
        return gmsh_elem_list.index((rank, node_count))
