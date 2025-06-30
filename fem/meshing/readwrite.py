import os
import numpy as np
from warnings import warn

from myjive.fem import NodeSet, XNodeSet, ElementSet, XElementSet, ElementGroup

__all__ = ["write_mesh", "read_mesh", "get_gmsh_elem_info", "get_gmsh_elem_type"]


def write_mesh(mesh, fname):
    file, extension = os.path.splitext(fname)

    if extension == ".mesh":
        _write_manual(mesh, fname)
    elif extension == ".msh":
        _write_gmsh(mesh, fname)
    else:
        raise ValueError("Invalid file type passed to write_mesh")


def read_mesh(fname, *, read_groups=False):
    file, extension = os.path.splitext(fname)

    if extension == ".mesh":
        if read_groups:
            warn("Element groups cannot be read from manual mesh")
        return _read_manual(fname)
    elif extension == ".msh":
        return _read_gmsh(fname, read_groups=read_groups)
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
            line = str(node_id)
            for coord in coords:
                line += " " + str(coord)
            line += "\n"
            file.write(line)

        file.write("elements (node#1, node#2, [node#3, ...])\n")
        for ielem, inodes in enumerate(elems):
            node_ids = nodes.get_node_ids(inodes)
            line = str(node_ids[0])
            for node_id in node_ids[1:]:
                line += " " + str(node_ids)
            line += "\n"
            file.write(line)


def _write_gmsh(mesh, fname):
    if isinstance(mesh, ElementSet):
        elems = mesh
        nodes = elems.get_nodes()
        write_groups = False
    elif isinstance(mesh, tuple):
        nodes = mesh[0]
        elems = mesh[1]
        if len(mesh) > 2:
            egroups = mesh[2]
            write_groups = True
        else:
            write_groups = False

    assert isinstance(nodes, NodeSet)
    assert isinstance(elems, ElementSet)

    # nodes = elems.get_nodes()
    rank = nodes.rank()
    rank3coords = np.zeros(3)

    if write_groups:
        group_name_map = {}
        for i, key in enumerate(egroups.keys()):
            group_name_map[key] = i + 1

    path, file = os.path.split(fname)
    if len(path) > 0 and not os.path.isdir(path):
        os.makedirs(path)

    with open(fname, "w") as file:
        file.write("$MeshFormat\n")
        file.write("2.2 0 8\n")
        file.write("$EndMeshFormat\n")

        if write_groups:
            file.write("$PhysicalNames\n")
            file.write(str(len(group_name_map)) + "\n")
            for name, index in group_name_map.items():
                file.write("2 " + str(index) + ' "' + name + '"\n')
            file.write("$EndPhysicalNames\n")

        file.write("$Nodes\n")
        file.write("{}\n".format(len(nodes)))
        for inode, coords in enumerate(nodes):
            node_id = nodes.get_node_id(inode)
            rank3coords[:rank] = coords
            line = str(node_id)
            line += " " + str(rank3coords[0])
            line += " " + str(rank3coords[1])
            line += " " + str(rank3coords[2])
            line += "\n"
            file.write(line)
        file.write("$EndNodes\n")

        file.write("$Elements\n")
        file.write("{}\n".format(len(elems)))
        for ielem, inodes in enumerate(elems):
            node_ids = nodes.get_node_ids(inodes)
            elem_id = elems.get_elem_id(ielem)
            elem_type = get_gmsh_elem_type(rank, len(inodes))
            line = str(elem_id) + " " + str(elem_type)

            if write_groups:
                match = ""
                for name, egroup in egroups.items():
                    if ielem in egroup:
                        assert match == ""
                        match = name
                assert match != ""
                idx = str(group_name_map[match])
                line += " 2 " + idx + " " + idx
            else:
                line += " 2 1 1"

            for node_id in node_ids:
                line += " " + str(node_id)
            line += "\n"
            file.write(line)
        file.write("$EndElements\n")


def _read_gmsh(fname, *, read_groups=False):
    nodes = XNodeSet()
    elems = XElementSet(nodes)

    parse_nodes = False
    parse_elems = False

    if read_groups:
        groups = {}
        group_name_map = {}
        parse_names = False

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

            if read_groups:
                if line == "$PhysicalNames\n":
                    parse_names = True
                elif line == "$EndPhysicalNames\n":
                    parse_names = False

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

                if read_groups:
                    ielem = len(elems) - 1
                    group_tag = int(sp[3])

                    if group_tag not in groups:
                        groups[group_tag] = ElementGroup(elems)

                    groups[group_tag].add_element(ielem)

            if read_groups and parse_names and len(sp) > 1:
                group_tag = int(sp[1])
                group_name = sp[2].strip('"')
                group_name_map[group_tag] = group_name

        mesh_rank = np.max(ranks)
        coords = nodes.get_coords()
        if coords.shape[1] > mesh_rank:
            nodes.set_coords(coords[:, :mesh_rank])

    nodes.to_nodeset()
    elems.to_elementset()

    if read_groups:
        named_groups = {}

        for group_tag, group in groups.items():
            group_name = group_name_map.get(group_tag, "gmsh" + str(group_tag))
            named_groups[group_name] = group

        return nodes, elems, named_groups
    else:
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
