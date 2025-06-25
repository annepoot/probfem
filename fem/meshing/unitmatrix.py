import os
import numpy as np
from scipy.sparse import csr_array
import ctypes as ct

from fem.meshing import (
    create_bboxes,
    list_point_bbox_intersections,
    check_point_in_shape,
)
from fem.jive import libcppbackend
from fem.jive.ctypesutils import (
    to_ctypes,
    to_numpy,
    DOUBLE_ARRAY_PTR,
    LONG_ARRAY_PTR,
    POINTSET_PTR,
    GROUPSET_PTR,
)

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


def create_phi_matrix_t3_cpp(
    coarse_elems, fine_elems, coarse_dofs, fine_dofs, coarse_shape
):
    coarse_nodes = coarse_elems.get_nodes()
    fine_nodes = fine_elems.get_nodes()

    buffer_size = 10 * len(fine_nodes)
    Phi_rowidx = np.zeros(buffer_size, dtype=int)
    Phi_colidx = np.zeros(buffer_size, dtype=int)
    Phi_values = np.zeros(buffer_size, dtype=float)

    ct_phi_rowidx = to_ctypes(Phi_rowidx)
    ct_phi_colidx = to_ctypes(Phi_colidx)
    ct_phi_values = to_ctypes(Phi_values)

    cnd = coarse_nodes._data[: len(coarse_nodes)].copy()
    ced = coarse_elems._data[: len(coarse_elems)].copy()
    ces = coarse_elems._groupsizes[: len(coarse_elems)].copy()
    fnd = fine_nodes._data[: len(fine_nodes)].copy()
    fed = fine_elems._data[: len(fine_elems)].copy()
    fes = fine_elems._groupsizes[: len(fine_elems)].copy()

    ct_coarse_nodes = POINTSET_PTR(to_ctypes(cnd))
    ct_coarse_elems = GROUPSET_PTR(to_ctypes(ced), to_ctypes(ces))
    ct_fine_nodes = POINTSET_PTR(to_ctypes(fnd))
    ct_fine_elems = GROUPSET_PTR(to_ctypes(fed), to_ctypes(fes))

    cpp_func = libcppbackend.create_phi_matrix_t3
    cpp_func.arg_types = (
        LONG_ARRAY_PTR,  # Phi_rowidx
        LONG_ARRAY_PTR,  # Phi_colidx
        DOUBLE_ARRAY_PTR,  # Phi_values
        POINTSET_PTR,  # coarse_nodes
        GROUPSET_PTR,  # coarse_elems
        POINTSET_PTR,  # fine_nodes
        GROUPSET_PTR,  # fine_elems
    )
    cpp_func.res_type = ct.c_int
    phi_size = cpp_func(
        ct_phi_rowidx,
        ct_phi_colidx,
        ct_phi_values,
        ct_coarse_nodes,
        ct_coarse_elems,
        ct_fine_nodes,
        ct_fine_elems,
    )

    Phi_rowidx = to_numpy(ct_phi_rowidx)[:phi_size]
    Phi_colidx = to_numpy(ct_phi_colidx)[:phi_size]
    Phi_values = to_numpy(ct_phi_values)[:phi_size]

    # Combine indices to unique keys
    coords = np.stack((Phi_rowidx, Phi_colidx), axis=1)
    _, unique_indices = np.unique(coords, axis=0, return_index=True)

    Phi_rowidx = Phi_rowidx[unique_indices]
    Phi_colidx = Phi_colidx[unique_indices]
    Phi_values = Phi_values[unique_indices]

    Phi_nodal = csr_array(
        (Phi_values, (Phi_rowidx, Phi_colidx)),
        shape=(len(fine_nodes), len(coarse_nodes)),
    )

    dof_types = fine_dofs.get_types()
    assert dof_types == coarse_dofs.get_types()
    dof_type_count = len(dof_types)
    dof_count_f = Phi_nodal.shape[0] * dof_type_count
    dof_count_c = Phi_nodal.shape[1] * dof_type_count

    phi_nnz = Phi_nodal.nnz * dof_type_count
    phi_data = np.zeros(phi_nnz, dtype=float)
    phi_rows = np.zeros(phi_nnz, dtype=int)
    phi_cols = np.zeros(phi_nnz, dtype=int)

    phi_nodal_rows, phi_nodal_cols = Phi_nodal.nonzero()
    phi_rows = fine_dofs.get_dofs(phi_nodal_rows, dof_types)
    phi_cols = coarse_dofs.get_dofs(phi_nodal_cols, dof_types)
    phi_data = np.repeat(Phi_nodal.data, dof_type_count)

    Phi = csr_array((phi_data, (phi_rows, phi_cols)), shape=(dof_count_f, dof_count_c))

    assert np.allclose(np.sum(Phi, axis=1), 1)

    return Phi


def create_phi_matrix(coarse_elems, fine_elems, coarse_dofs, fine_dofs, coarse_shape):
    coarse_nodes = coarse_elems.get_nodes()
    fine_nodes = fine_elems.get_nodes()

    dof_types = fine_dofs.get_types()
    assert dof_types == coarse_dofs.get_types()

    Phi = np.zeros((fine_dofs.dof_count(), coarse_dofs.dof_count()))

    coarse_bboxes = create_bboxes(coarse_elems)

    tol = 1e-8

    for inode_f, coords_f in enumerate(fine_nodes):
        idof_f = fine_dofs.get_dof(inode_f, "dx")
        ielems_c = list_point_bbox_intersections(coords_f, coarse_bboxes, tol=tol)

        for ielem_c in ielems_c:
            inodes_c = coarse_elems[ielem_c]
            coords_c = coarse_nodes.get_some_coords(inodes_c)

            if check_point_in_shape(coords_f, coords_c, tol=tol):
                loc_point = coarse_shape.get_local_point(coords_f, coords_c)
                sfunc_evals = coarse_shape.eval_shape_functions(loc_point)

                for dof_type in dof_types:
                    idof_f = fine_dofs.get_dof(inode_f, dof_type)
                    idofs_c = coarse_dofs.get_dofs(inodes_c, [dof_type])
                    Phi[idof_f, idofs_c] = sfunc_evals

    Phi = csr_array(Phi)

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

    if shapec.node_count() == 3 and shapec.global_rank() == 2:
        Phi = create_phi_matrix_t3_cpp(elemsc, elemsf, dofsc, dofsf, shapec)
    else:
        Phi = create_phi_matrix(elemsc, elemsf, dofsc, dofsf, shapec)

    return Phi
