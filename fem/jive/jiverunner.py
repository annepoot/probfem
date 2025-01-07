import os
import numpy as np
import ctypes as ct
from scipy.sparse import csr_array

from myjive.app import main
from myjivex.declare import declare_all
from myjive.solver import Constraints

__all__ = ["MyJiveRunner", "CJiveRunner"]


class MyJiveRunner:

    def __init__(self, props):
        self.props = props

    def __call__(self):
        globdat = main.jive(self.props, extra_declares=[declare_all])
        return globdat


class CJiveRunner:

    def __init__(self, fname, node_count, rank):
        self.fname = fname.encode("utf-8")
        self.node_count = node_count
        self.rank = rank

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
            DOUBLE_VEC_PTR((ct.c_double * dof_count)(), ct.c_int(dof_count)),  # state0
            DOUBLE_VEC_PTR((ct.c_double * dof_count)(), ct.c_int(dof_count)),  # intVector
            DOUBLE_VEC_PTR((ct.c_double * dof_count)(), ct.c_int(dof_count)),  # extVector
            SPARSE_MAT_PTR(
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
            DOUBLE_MAT_PTR(  # coords
                (ct.c_double * (self.node_count * self.rank))(),
                ct.c_int(self.node_count),
                ct.c_int(self.rank),
            ),
            INT_MAT_PTR(  # dofs
                (ct.c_int * (self.node_count * self.rank))(),
                ct.c_int(self.node_count),
                ct.c_int(self.rank),
            ),
            CONSTRAINTS_PTR(
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
        coords = to_numpy(globdat.coords)
        dofs = to_numpy(globdat.dofs)

        matrix0_values = to_numpy(globdat.matrix0.values)
        matrix0_indices = to_numpy(globdat.matrix0.indices)
        matrix0_offsets = to_numpy(globdat.matrix0.offsets)

        matrix0 = csr_array((matrix0_values, matrix0_indices, matrix0_offsets))

        cdofs = to_numpy(globdat.constraints.dofs)
        cvals = to_numpy(globdat.constraints.values)

        constraints = Constraints()
        for cdof, cval in zip(cdofs, cvals):
            constraints.add_constraint(cdof, cval)

        return {
            "state0": state0,
            "intForce": intForce,
            "extForce": extForce,
            "matrix0": matrix0,
            "coords": coords,
            "dofs": dofs,
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


class GLOBDAT(ct.Structure):
    _fields_ = [
        ("state0", DOUBLE_VEC_PTR),
        ("intForce", DOUBLE_VEC_PTR),
        ("extForce", DOUBLE_VEC_PTR),
        ("matrix0", SPARSE_MAT_PTR),
        ("coords", DOUBLE_MAT_PTR),
        ("dofs", INT_MAT_PTR),
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
