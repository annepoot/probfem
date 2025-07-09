import numpy as np

from myjive.fem import NodeSet, XNodeSet, ElementSet, XElementSet


def u_exact(x, *, k, E, f):
    nu = np.sqrt(k / E)
    eps = f / E

    A = eps / (nu * (np.exp(nu) - np.exp(-nu)))

    return A * (np.exp(nu * x) + np.exp(-nu * x))


def invert_mesh(mesh):
    if isinstance(mesh, ElementSet):
        elems = mesh
        nodes = elems.get_nodes()
    else:
        nodes, elems = mesh
    assert isinstance(nodes, NodeSet)
    assert isinstance(elems, ElementSet)

    coords = nodes.get_coords()

    left_boundary = np.min(coords, axis=0)
    right_boundary = np.max(coords, axis=0)

    inv_nodes = XNodeSet()
    inv_nodes.add_node(left_boundary)
    for inodes in elems:
        midpoint = np.mean(nodes[inodes], axis=0)
        inv_nodes.add_node(midpoint)
    inv_nodes.add_node(right_boundary)
    inv_nodes.to_nodeset()

    inv_coords = inv_nodes.get_coords()
    sort_idx = np.argsort(inv_coords[:, 0], axis=0)

    inv_elems = XElementSet(inv_nodes)
    for ielem in np.arange(len(inv_nodes) - 1):
        inodes = np.array([sort_idx[ielem], sort_idx[ielem + 1]])
        inv_elems.add_element(inodes)
    inv_elems.to_elementset()

    return inv_nodes, inv_elems


def random_mesh(*, n, seed):
    rng = np.random.default_rng(seed)
    coords = np.zeros((n + 1, 1))
    coords[0, 0] = 0.0
    coords[n, 0] = 1.0
    coords[1:n, 0] = np.sort(rng.uniform(size=n - 1))

    nodes = XNodeSet()
    for coord in coords:
        nodes.add_node(coord)
    nodes.to_nodeset()

    elems = XElementSet(nodes)
    for ielem in np.arange(len(nodes) - 1):
        inodes = np.array([ielem, ielem + 1])
        elems.add_element(inodes)
    elems.to_elementset()

    return nodes, elems
