import numpy as np
import ctypes as ct

PTR = ct.POINTER


class INT_VEC_PTR(ct.Structure):
    _fields_ = [
        ("ptr", PTR(ct.c_int)),
        ("size", ct.c_int),
    ]


class DOUBLE_VEC_PTR(ct.Structure):
    _fields_ = [
        ("ptr", PTR(ct.c_double)),
        ("size", ct.c_int),
    ]


class STRING_PTR(ct.Structure):
    _fields_ = [
        ("ptr", PTR(ct.c_char)),
        ("size", ct.c_int),
    ]


class INT_MAT_PTR(ct.Structure):
    _fields_ = [
        ("ptr", PTR(ct.c_int)),
        ("size0", ct.c_int),
        ("size1", ct.c_int),
    ]


class DOUBLE_MAT_PTR(ct.Structure):
    _fields_ = [
        ("ptr", PTR(ct.c_double)),
        ("size0", ct.c_int),
        ("size1", ct.c_int),
    ]


class SPARSE_MAT_PTR(ct.Structure):
    _fields_ = [
        ("values", DOUBLE_VEC_PTR),
        ("indices", INT_VEC_PTR),
        ("offsets", INT_VEC_PTR),
    ]


class SHAPE_PTR(ct.Structure):
    _fields_ = [
        ("type", STRING_PTR),
        ("ischeme", STRING_PTR),
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
        ("shape", SHAPE_PTR),
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


def initialize_globdat(*, node_count, elem_count, rank, max_elem_node_count):
    dof_count = node_count * rank

    globdat = GLOBDAT(
        POINTSET_PTR(  # nodeSet
            DOUBLE_MAT_PTR(  # nodeSet.data
                (ct.c_double * (node_count * rank))(),
                ct.c_int(node_count),
                ct.c_int(rank),
            ),
        ),
        GROUPSET_PTR(  # elemSet
            INT_MAT_PTR(  # elemSet.data
                (ct.c_int * (elem_count * max_elem_node_count))(),
                ct.c_int(elem_count),
                ct.c_int(max_elem_node_count),
            ),
            INT_VEC_PTR(  # elemSet.sizes
                (ct.c_int * elem_count)(),
                ct.c_int(elem_count),
            ),
        ),
        INT_MAT_PTR(  # dofSpace
            (ct.c_int * (node_count * rank))(),
            ct.c_int(node_count),
            ct.c_int(rank),
        ),
        DOUBLE_VEC_PTR((ct.c_double * dof_count)(), ct.c_int(dof_count)),  # state0
        DOUBLE_VEC_PTR((ct.c_double * dof_count)(), ct.c_int(dof_count)),  # intVector
        DOUBLE_VEC_PTR((ct.c_double * dof_count)(), ct.c_int(dof_count)),  # extVector
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
        SHAPE_PTR(  # shape
            STRING_PTR(  # shape.type
                (ct.c_char * 64)(),
                ct.c_int(64),
            ),
            STRING_PTR(  # shape.ischeme
                (ct.c_char * 64)(),
                ct.c_int(64),
            ),
        ),
    )

    return globdat
