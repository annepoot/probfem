import os
import numpy as np
import gmsh

from myjive.fem import XNodeSet, XElementSet

from fem.meshing.readwrite import get_gmsh_elem_info


def create_mesh(*, h, L, H, x, y, a, theta, r_rel, h_meas, n_refine=0, tol=1e-8):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.Verbosity", 2)  # only print errors and warnings

    gmsh.model.add("example.mesh")

    occ = gmsh.model.occ

    points = []
    points_with_dim = []
    for x_point in np.linspace(0, L, int(L / h_meas) + 1):
        for y_point in [0.0, H]:
            p = occ.addPoint(x_point, y_point, 0.0)
            points.append(p)
            points_with_dim.append((0, p))

    for y_point in np.linspace(0, H, int(H / h_meas) + 1)[1:-1]:
        x_point = L
        p = occ.addPoint(x_point, y_point, 0.0)
        points.append(p)
        points_with_dim.append((0, p))

    main_rect = occ.addRectangle(0.0, 0.0, 0.0, L, H)
    r = a * r_rel
    hole_rect = occ.addRectangle(x - 0.5 * a, y - 0.5 * a, 0.0, a, a, roundedRadius=r)

    main_with_dim = [(2, main_rect)]
    hole_with_dim = [(2, hole_rect)]

    occ.rotate(hole_with_dim, x, y, 0.0, 0.0, 0.0, 1.0, theta)
    diff = occ.cut(main_with_dim, hole_with_dim)[0]
    occ.fragment(points_with_dim, main_with_dim)[0]

    # Generate mesh
    occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [diff[0][1]])

    gmsh.option.setNumber("Mesh.MeshSizeMin", h)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)

    gmsh.model.mesh.generate(2)
    nodes, elems = get_nodes_and_elems(gmsh)
    output = ((nodes, elems),)

    # Refine the mesh n times
    for i in range(n_refine):
        gmsh.refine()
        nodes, elems = get_nodes_and_elems(gmsh)
        output += ((nodes, elems),)

    gmsh.model.remove()
    gmsh.finalize()

    return output


def get_nodes_and_elems(gmsh):
    node_ids, node_coords, _ = gmsh.model.mesh.getNodes()
    rank = max([i for i in range(3) if np.any(node_coords[i::3])]) + 1
    node_coords = np.reshape(node_coords, (-1, 3))[:, :rank]

    nodes = XNodeSet()
    nodes.add_nodes(node_coords, node_ids)
    nodes.to_nodeset()

    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
    assert len(elem_types) == 1
    elem_type = elem_types[0]
    rank, node_count = get_gmsh_elem_info(elem_type)
    elem_ids = elem_tags[0]
    elem_inodes = nodes.find_items(elem_node_tags[0])
    elem_inodes = np.reshape(elem_inodes, (-1, node_count))
    elem_sizes = np.zeros(len(elem_ids), dtype=int) + node_count

    elems = XElementSet(nodes)
    elems.add_elements(elem_inodes, elem_sizes, elem_ids)
    elems.to_elementset()

    return nodes, elems
