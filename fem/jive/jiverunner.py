import os
import numpy as np
import ctypes as ct
from scipy.sparse import csr_array

from myjive.app import main
from myjivex.declare import declare_all
from myjive.fem import XNodeSet, XElementSet, DofSpace
from myjive.solver import Constraints

__all__ = ["MyJiveRunner", "CJiveRunner"]


class MyJiveRunner:

    def __init__(self, props):
        self.props = props

    def __call__(self):
        globdat = main.jive(self.props, extra_declares=[declare_all])
        return globdat


class CJiveRunner:

    def __init__(self, fname, *, node_count, elem_count, rank, max_elem_node_count):
        self.fname = fname.encode("utf-8")
        self.node_count = node_count
        self.elem_count = elem_count
        self.rank = rank
        self.max_elem_node_count = max_elem_node_count

    def __call__(self):
        PTR = ct.POINTER

        loader = ct.LibraryLoader(ct.CDLL)
        abspath = os.path.abspath(os.path.join(__file__, "..", "src", "liblinear.so"))
        liblinear = loader.LoadLibrary(abspath)

        globdat_func = liblinear.getGlobdat
        globdat_func.argtypes = (
            PTR(GLOBDAT),
            PTR(ct.c_char),
        )

        dof_count = self.node_count * self.rank

        globdat = GLOBDAT(
            POINTSET_PTR(  # nodeSet
                DOUBLE_MAT_PTR(  # nodeSet.data
                    (ct.c_double * (self.node_count * self.rank))(),
                    ct.c_int(self.node_count),
                    ct.c_int(self.rank),
                ),
            ),
            GROUPSET_PTR(  # elemSet
                INT_MAT_PTR(  # elemSet.data
                    (ct.c_int * (self.elem_count * self.max_elem_node_count))(),
                    ct.c_int(self.elem_count),
                    ct.c_int(self.max_elem_node_count),
                ),
                INT_VEC_PTR(  # elemSet.sizes
                    (ct.c_int * self.elem_count)(),
                    ct.c_int(self.elem_count),
                ),
            ),
            INT_MAT_PTR(  # dofSpace
                (ct.c_int * (self.node_count * self.rank))(),
                ct.c_int(self.node_count),
                ct.c_int(self.rank),
            ),
            DOUBLE_VEC_PTR((ct.c_double * dof_count)(), ct.c_int(dof_count)),  # state0
            DOUBLE_VEC_PTR(
                (ct.c_double * dof_count)(), ct.c_int(dof_count)
            ),  # intVector
            DOUBLE_VEC_PTR(
                (ct.c_double * dof_count)(), ct.c_int(dof_count)
            ),  # extVector
            SPARSE_MAT_PTR(  # matrix0
                DOUBLE_VEC_PTR(  # matrix0.values
                    (ct.c_double * (dof_count * 20))(),
                    ct.c_int(dof_count * 20),
                ),
                INT_VEC_PTR(  # matrix0.indices
                    (ct.c_int * (dof_count * 20))(), ct.c_int(dof_count * 20)
                ),
                INT_VEC_PTR(  # matrix0.offsets
                    (ct.c_int * (dof_count + 1))(), ct.c_int(dof_count + 1)
                ),
            ),
            CONSTRAINTS_PTR(  # constraints
                INT_VEC_PTR(  # constraints.dofs
                    (ct.c_int * dof_count)(),
                    ct.c_int(dof_count),
                ),
                DOUBLE_VEC_PTR(  # constraints.values
                    (ct.c_double * dof_count)(),
                    ct.c_int(dof_count),
                ),
            ),
        )

        globdat_func(
            ct.byref(globdat),
            self.fname,
        )

        state0 = to_numpy(globdat.state0)
        intForce = to_numpy(globdat.intForce)
        extForce = to_numpy(globdat.extForce)
        dofs_data = to_numpy(globdat.dofSpace)

        nodes_data = to_numpy(globdat.nodeSet.data)
        elems_data = to_numpy(globdat.elemSet.data)
        elems_sizes = to_numpy(globdat.elemSet.sizes)

        matrix0_values = to_numpy(globdat.matrix0.values)
        matrix0_indices = to_numpy(globdat.matrix0.indices)
        matrix0_offsets = to_numpy(globdat.matrix0.offsets)

        matrix0 = csr_array((matrix0_values, matrix0_indices, matrix0_offsets))

        cdofs = to_numpy(globdat.constraints.dofs)
        cvals = to_numpy(globdat.constraints.values)

        constraints = Constraints()
        for cdof, cval in zip(cdofs, cvals):
            constraints.add_constraint(cdof, cval)

        nodes = XNodeSet()
        for coord in nodes_data:
            nodes.add_node(coord)
        nodes.to_nodeset()

        elems = XElementSet(nodes)
        for inodes, elem_node_count in zip(elems_data, elems_sizes):
            elems.add_element(inodes[:elem_node_count])
        elems.to_elementset

        dofs = DofSpace()
        for it in range(dofs_data.shape[1]):
            dof_type = ["dx", "dy", "dz"][it]
            dofs.add_type(dof_type)

            for inode in range(dofs_data.shape[0]):
                dofs._dofs[dof_type][inode] = dofs_data[inode, it]
                
        dofs._count = dofs_data.shape[0] * dofs_data.shape[1]

        return {
            "nodeSet": nodes,
            "elemSet": elems,
            "dofSpace": dofs,
            "state0": state0,
            "intForce": intForce,
            "extForce": extForce,
            "matrix0": matrix0,
            "constraints": constraints,
        }


class INT_VEC_PTR(ct.Structure):
    _fields_ = [
        ("ptr", ct.POINTER(ct.c_int)),
        ("size", ct.c_int),
    ]


class DOUBLE_VEC_PTR(ct.Structure):
    _fields_ = [
        ("ptr", ct.POINTER(ct.c_double)),
        ("size", ct.c_int),
    ]


class INT_MAT_PTR(ct.Structure):
    _fields_ = [
        ("ptr", ct.POINTER(ct.c_int)),
        ("size0", ct.c_int),
        ("size1", ct.c_int),
    ]


class DOUBLE_MAT_PTR(ct.Structure):
    _fields_ = [
        ("ptr", ct.POINTER(ct.c_double)),
        ("size0", ct.c_int),
        ("size1", ct.c_int),
    ]


class SPARSE_MAT_PTR(ct.Structure):
    _fields_ = [
        ("values", DOUBLE_VEC_PTR),
        ("indices", INT_VEC_PTR),
        ("offsets", INT_VEC_PTR),
    ]


class CONSTRAINTS_PTR(ct.Structure):
    _fields_ = [
        ("dofs", INT_VEC_PTR),
        ("values", DOUBLE_VEC_PTR),
    ]


class POINTSET_PTR(ct.Structure):
    _fields_ = [
        ("data", DOUBLE_MAT_PTR),
    ]


class GROUPSET_PTR(ct.Structure):
    _fields_ = [
        ("data", INT_MAT_PTR),
        ("sizes", INT_VEC_PTR),
    ]


class GLOBDAT(ct.Structure):
    _fields_ = [
        ("nodeSet", POINTSET_PTR),
        ("elemSet", GROUPSET_PTR),
        ("dofSpace", INT_MAT_PTR),
        ("state0", DOUBLE_VEC_PTR),
        ("intForce", DOUBLE_VEC_PTR),
        ("extForce", DOUBLE_VEC_PTR),
        ("matrix0", SPARSE_MAT_PTR),
        ("constraints", CONSTRAINTS_PTR),
    ]


def to_numpy(c_obj):
    if ("size", ct.c_int) in c_obj._fields_:
        shape = (c_obj.size,)
    else:
        shape = tuple()
        for i in range(4):
            size_i = "size" + str(i)
            if (size_i, ct.c_int) in c_obj._fields_:
                shape += (getattr(c_obj, size_i),)
            else:
                break

    return np.ctypeslib.as_array(c_obj.ptr, shape).copy()
