import os
import numpy as np
import ctypes as ct

from fem.jive import libcppbackend
from fem.meshing import get_patches_around_nodes


__all__ = ["calc_perturbed_coords"]

loader = ct.LibraryLoader(ct.CDLL)
abspath = os.path.abspath(os.path.join(__file__, "..", "libperturbation.so"))
libpermutation = loader.LoadLibrary(abspath)

PTR = ct.POINTER


def calc_perturbed_coords(
    *,
    ref_coords,
    elems,
    elem_sizes,
    p,
    boundary,
    rng,
    omit_nodes=[],
    tol=1e-8,
    patches=None
):
    h = np.max(elem_sizes)
    node_count = ref_coords.shape[0]
    rank = ref_coords.shape[1]
    pert_coords = np.copy(ref_coords)

    # rng generation beforehand is more efficient
    std_uniform = rng.uniform(size=(node_count, rank))

    if patches is None:
        patches = get_patches_around_nodes(elems)

    for inode, coords in enumerate(pert_coords):
        if inode in omit_nodes:
            continue

        if rank == 1:
            alpha_i_bar = std_uniform[inode] - 0.5
        elif rank == 2:
            r = 0.25 * np.sqrt(std_uniform[inode, 0])
            theta = std_uniform[inode, 1] * 2 * np.pi
            alpha_i_bar = r * np.array([np.cos(theta), np.sin(theta)])
        else:
            raise NotImplementedError(
                "RandomMeshModel has not been implemented for 3D yet"
            )

        patch = patches[inode]
        h_i_bar = np.min(elem_sizes[patch])
        alpha_i = (h_i_bar / h) ** p * alpha_i_bar

        if inode in boundary:
            if rank == 1:
                alpha_i *= 0.0

            elif rank == 2:
                norm1, norm2 = boundary[inode]

                if abs(norm1[0] * norm2[1] - norm1[1] * norm2[0]) < tol:
                    basis = np.array([-norm1[1], norm2[0]])
                    alpha_i = np.dot(alpha_i, basis) * basis
                else:
                    alpha_i *= 0.0
            else:
                assert False

        coords += h**p * alpha_i
        pert_coords[inode] = coords

    return pert_coords


def calc_perturbed_coords_cpp(
    *,
    ref_coords,
    elems,
    elem_sizes,
    p,
    boundary,
    rng,
    omit_nodes=[],
    tol=1e-8,
    patches=None
):
    node_count = ref_coords.shape[0]
    rank = ref_coords.shape[1]
    pert_coords = np.copy(ref_coords)

    # rng generation beforehand is more efficient
    std_uniform = rng.uniform(size=(node_count, rank))

    if patches is None:
        patches = get_patches_around_nodes(elems)

    omit_nodes_arr = np.array(omit_nodes, dtype=np.long)

    pert_coords_ptr = pert_coords.ctypes.data_as(PTR(ct.c_double))
    pert_coords_shape = pert_coords.ctypes.shape_as(ct.c_long)
    elem_sizes_ptr = elem_sizes.ctypes.data_as(PTR(ct.c_double))
    elem_sizes_shape = elem_sizes.ctypes.shape_as(ct.c_long)
    std_uniform_ptr = std_uniform.ctypes.data_as(PTR(ct.c_double))
    std_uniform_shape = std_uniform.ctypes.shape_as(ct.c_long)
    omit_nodes_ptr = omit_nodes_arr.ctypes.data_as(PTR(ct.c_long))
    omit_nodes_shape = omit_nodes_arr.ctypes.shape_as(ct.c_long)

    patches_ptr_list = []
    patches_size_list = []
    for patch in patches:
        patch_size = len(patch)
        patch_ptr = (ct.c_long * patch_size)(*patch)
        patches_ptr_list.append(patch_ptr)
        patches_size_list.append(patch_size)

    patches_size = len(patches)
    patches_patch_ptr = (PTR(ct.c_long) * patches_size)(*patches_ptr_list)
    patches_patch_size = (ct.c_long * patches_size)(*patches_size_list)

    boundary_nodes_arr = np.array(list(boundary.keys()))
    boundary_normals_arr = np.array(list(boundary.values()))
    boundary_nodes_ptr = boundary_nodes_arr.ctypes.data_as(PTR(ct.c_long))
    boundary_nodes_shape = boundary_nodes_arr.ctypes.shape_as(ct.c_long)
    boundary_normals_ptr = boundary_normals_arr.ctypes.data_as(PTR(ct.c_double))
    boundary_normals_shape = boundary_normals_arr.ctypes.shape_as(ct.c_long)

    calc_perturbed_coords_func = libcppbackend.calc_perturbed_coords
    calc_perturbed_coords_func.argtypes = (
        PTR(ct.c_double),  # pert_coords_ptr,
        PTR(ct.c_long),  # pert_coords_shape,
        ct.c_double,  # p,
        ct.c_double,  # tol,
        PTR(ct.c_double),  # elem_sizes_ptr,
        PTR(ct.c_long),  # elem_sizes_shape,
        PTR(ct.c_double),  # std_uniform_ptr,
        PTR(ct.c_long),  # std_uniform_shape,
        PTR(PTR(ct.c_long)),  # patches_patch_ptr,
        PTR(ct.c_long),  # patches_patch_size,
        ct.c_long,  # patches_size,
        PTR(ct.c_long),  # boundary_nodes_ptr,
        PTR(ct.c_long),  # boundary_nodes_shape,
        PTR(ct.c_double),  # boundary_normals_ptr,
        PTR(ct.c_long),  # boundary_normals_shape,
        PTR(ct.c_long),  # omit_nodes_ptr,
        PTR(ct.c_long),  # omit_nodes_shape,
    )
    calc_perturbed_coords_func.restype = ct.c_void_p

    calc_perturbed_coords_func(
        pert_coords_ptr,
        pert_coords_shape,
        p,
        tol,
        elem_sizes_ptr,
        elem_sizes_shape,
        std_uniform_ptr,
        std_uniform_shape,
        patches_patch_ptr,
        patches_patch_size,
        patches_size,
        boundary_nodes_ptr,
        boundary_nodes_shape,
        boundary_normals_ptr,
        boundary_normals_shape,
        omit_nodes_ptr,
        omit_nodes_shape,
    )

    return pert_coords
