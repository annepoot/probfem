import numpy as np
import gmsh

from myjive.fem import XNodeSet, XElementSet

from fem.meshing.readwrite import get_gmsh_elem_info


def create_mesh(*, h, L, H, U, x, y, a, theta, r_rel, h_meas, n_refine=0, tol=1e-8):
    w_sup = 0.2
    h_sup = 0.1
    r = a * r_rel

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.Verbosity", 2)  # only print errors and warnings

    gmsh.model.add("example.mesh")

    occ = gmsh.model.occ

    points = []
    point_tags_with_dim = []

    for obs_loc in get_observation_locations(L=L, H=H, h_meas=h_meas):
        p = occ.addPoint(obs_loc[0], obs_loc[1], 0.0)
        points.append(p)
        point_tags_with_dim.append((0, p))

    # left support
    p = occ.addPoint(U, -h_sup, 0.0)
    points.append(p)
    point_tags_with_dim.append((0, p))

    # midpoint
    p = occ.addPoint(L / 2, H + h_sup, 0.0)
    points.append(p)
    point_tags_with_dim.append((0, p))

    # right support
    p = occ.addPoint(L - U, -h_sup, 0.0)
    points.append(p)
    point_tags_with_dim.append((0, p))

    main_rect = occ.addRectangle(0.0, 0.0, 0.0, L, H)
    hole_rect = occ.addRectangle(x - 0.5 * a, y - 0.5 * a, 0.0, a, a, roundedRadius=r)
    left_rect = occ.addRectangle(U - 0.5 * w_sup, -h_sup, 0.0, w_sup, h_sup)
    right_rect = occ.addRectangle(L - U - 0.5 * w_sup, -h_sup, 0.0, w_sup, h_sup)
    mid_rect = occ.addRectangle(0.5 * L - 0.5 * w_sup, H, 0.0, w_sup, h_sup)

    beam_tags_with_dim = [(2, main_rect)]
    hole_tags_with_dim = [(2, hole_rect)]
    support_tags_with_dim = [(2, left_rect), (2, right_rect), (2, mid_rect)]

    occ.rotate(hole_tags_with_dim, x, y, 0.0, 0.0, 0.0, 1.0, theta)

    # First fragments supports with main rectangle
    sup_beam_tags_with_dim = occ.fragment(beam_tags_with_dim, support_tags_with_dim)[0]

    # Second fragment points with main rectangle
    fragments = occ.fragment(sup_beam_tags_with_dim, point_tags_with_dim)[0]
    sup_beam_tags_with_dim = [f for f in fragments if f[0] == 2]

    # Finally, cut the hole from the combined shape
    sup_hole_beam_tags_with_dim = occ.cut(sup_beam_tags_with_dim, hole_tags_with_dim)[0]

    # Generate mesh
    occ.synchronize()

    beam_tags = []
    support_tags = []

    for dim, tag in sup_hole_beam_tags_with_dim:
        if dim == 2:
            com = occ.getCenterOfMass(dim, tag)

            if com[1] < 0 or com[1] > H:
                support_tags.append(tag)
            else:
                beam_tags.append(tag)

    gmsh.model.addPhysicalGroup(2, beam_tags)
    gmsh.model.addPhysicalGroup(2, support_tags)

    gmsh.option.setNumber("Mesh.MeshSizeMin", h)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)

    gmsh.model.mesh.generate(2)
    nodes, elems = get_nodes_and_elems(gmsh)

    if n_refine > 0:
        output = ((nodes, elems),)
    else:
        output = (nodes, elems)

    # Refine the mesh n times
    for i in range(n_refine):
        gmsh.model.mesh.refine()
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


def get_observation_locations(*, L, H, h_meas):
    i = 0
    n_obs = int(2 * (L + H) / h_meas + 0.5)
    points = np.zeros((n_obs, 2))

    # observations at top and bottom edge
    for x_point in np.linspace(0, L, int(L / h_meas) + 1):
        for y_point in [0.0, H]:
            points[i, 0] = x_point
            points[i, 1] = y_point
            i = i + 1

    # observations at left and right edge
    for y_point in np.linspace(0, H, int(H / h_meas) + 1)[1:-1]:
        for x_point in [0.0, L]:
            points[i, 0] = x_point
            points[i, 1] = y_point
            i = i + 1

    assert i == n_obs

    return points
