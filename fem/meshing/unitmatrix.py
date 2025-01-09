import numpy as np
from scipy.sparse import csr_array

__all__ = ["create_unit_mass_matrix", "create_phi_matrix", "create_phi_from_globdat"]


def create_empty_matrix(elems, dofs, *, lumped):
    doftypes = dofs.get_types()
    dc = dofs.dof_count()

    if lumped:
        rowindices = [i for i in range(dc)]
        colindices = [i for i in range(dc)]

    else:
        rowindices = []
        colindices = []

        for inodes in elems:
            idofs = dofs.get_dofs(inodes, doftypes)

            for row in idofs:
                for col in idofs:
                    rowindices.append(row)
                    colindices.append(col)

    assert len(rowindices) == len(colindices)
    values = np.zeros(len(rowindices))

    empty_mat = csr_array(
        (values, (rowindices, colindices)), shape=(dc, dc), dtype=float
    )
    return empty_mat


def create_unit_mass_matrix(elems, dofs, shape, *, sparse, lumped):
    nodes = elems.get_nodes()
    doftypes = dofs.get_types()
    rank = len(doftypes)
    dofcount = shape.node_count() * rank

    if sparse:
        M = create_empty_matrix(elems, dofs, lumped=lumped)
    else:
        M = np.zeros((len(nodes), len(nodes)))

    for ielem, inodes in enumerate(elems):
        idofs = dofs.get_dofs(inodes, doftypes)
        coords = nodes.get_some_coords(inodes)

        sfuncs = shape.get_shape_functions()
        weights = shape.get_integration_weights(coords)

        elmat = np.zeros((dofcount, dofcount))

        for sfunc, weight in zip(sfuncs, weights):
            N_elem = np.zeros((rank, dofcount))
            for i in range(rank):
                N_elem[i, i::rank] = sfunc
            elmat += weight * N_elem.T @ N_elem

        if lumped:
            M[idofs, idofs] += np.sum(elmat, axis=0)
        else:
            M[np.ix_(idofs, idofs)] += elmat

    return M


def create_phi_matrix(coarse_elems, fine_elems, coarse_dofs, fine_dofs, coarse_shape):
    coarse_nodes = coarse_elems.get_nodes()
    fine_nodes = fine_elems.get_nodes()

    dof_types = fine_dofs.get_types()
    assert dof_types == coarse_dofs.get_types()
    rank = len(dof_types)

    Phi = np.zeros((fine_dofs.dof_count(), coarse_dofs.dof_count()))

    # Go over the coarse mesh
    for inodesc in coarse_elems:
        coordsc = coarse_nodes.get_some_coords(inodesc)

        # Get the bounding box of the coarse element
        bbox = np.zeros((rank, 2))
        for i in range(rank):
            bbox[i, 0] = min(coordsc[:, i])
            bbox[i, 1] = max(coordsc[:, i])

        # Go over the fine mesh
        for inode, coords in enumerate(fine_nodes):
            # Check if the node falls inside the bounding box
            inside = True
            for i in range(rank):
                if coords[i] < bbox[i, 0] or coords[i] > bbox[i, 1]:
                    inside = False
                    break

            # If so, check if the node falls inside the shape itself
            if inside:
                loc_point = coarse_shape.get_local_point(coords, coordsc)
                inside = coarse_shape.contains_local_point(loc_point, tol=1e-8)

            # If so, add the relative shape function values to the Phi matrix
            if inside:
                svals = np.round(coarse_shape.eval_shape_functions(loc_point), 12)
                idofs = fine_dofs.get_dofs([inode], dof_types)

                for i, inodec in enumerate(inodesc):
                    sval = svals[i]
                    idofsc = coarse_dofs.get_dofs([inodec], dof_types)
                    Phi[idofs, idofsc] = sval

    assert np.allclose(np.sum(Phi, axis=1), 1)

    return Phi


def create_phi_from_globdat(coarse_globdat, fine_globdat):
    elemsc = coarse_globdat["elemSet"]
    elemsf = fine_globdat["elemSet"]
    dofsc = coarse_globdat["dofSpace"]
    dofsf = fine_globdat["dofSpace"]

    if "shape" in coarse_globdat:
        shapec = coarse_globdat["shape"]
        assert type(shapec) == type(fine_globdat["shape"])
    else:
        shape_type = coarse_globdat["meshShape"]
        assert shape_type == fine_globdat["meshShape"]
        shape_ischeme = "Gauss1"
        shapec = coarse_globdat["shapeFactory"].get_shape(shape_type, shape_ischeme)

    Phi = create_phi_matrix(elemsc, elemsf, dofsc, dofsf, shapec)
    return Phi
