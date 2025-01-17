import numpy as np
import ctypes as ct
from scipy.sparse import csr_array

from myjive.fem import NodeSet, XNodeSet, ElementSet, XElementSet, DofSpace
from myjive.solver import Constraints
from myjive.declare import declare_shapes

PTR = ct.POINTER


class LONG_ARRAY_PTR(ct.Structure):
    _fields_ = [
        ("ptr", PTR(ct.c_long)),
        ("shape", PTR(ct.c_long)),
        ("dim", ct.c_long),
    ]


class DOUBLE_ARRAY_PTR(ct.Structure):
    _fields_ = [
        ("ptr", PTR(ct.c_double)),
        ("shape", PTR(ct.c_long)),
        ("dim", ct.c_long),
    ]


class CHAR_ARRAY_PTR(ct.Structure):
    _fields_ = [
        ("ptr", PTR(ct.c_char)),
        ("shape", PTR(ct.c_long)),
        ("dim", ct.c_long),
    ]


class LONG_VEC_PTR(LONG_ARRAY_PTR):
    def __init__(self, *args, **kw):
        if len(args) == 0:
            super().__init__(*args, **kw)
        else:
            assert len(args) == 2
            assert args[0]._type_ is ct.c_long
            assert args[1]._length_ == 1
            super().__init__(*args, 1, **kw)


class LONG_MAT_PTR(LONG_ARRAY_PTR):
    def __init__(self, *args, **kw):
        if len(args) == 0:
            super().__init__(*args, **kw)
        else:
            assert len(args) == 2
            assert args[0]._type_ is ct.c_long
            assert args[1]._length_ == 2
            super().__init__(*args, 2, **kw)


class DOUBLE_VEC_PTR(DOUBLE_ARRAY_PTR):
    def __init__(self, *args, **kw):
        if len(args) == 0:
            super().__init__(*args, **kw)
        else:
            assert len(args) == 2
            assert args[0]._type_ is ct.c_double
            assert args[1]._length_ == 1
            super().__init__(*args, 1, **kw)


class DOUBLE_MAT_PTR(DOUBLE_ARRAY_PTR):
    def __init__(self, *args, **kw):
        if len(args) == 0:
            super().__init__(*args, **kw)
        else:
            assert len(args) == 2
            assert args[0]._type_ is ct.c_double
            assert args[1]._length_ == 2
            super().__init__(*args, 2, **kw)


class STRING_PTR(CHAR_ARRAY_PTR):
    def __init__(self, *args, **kw):
        if len(args) == 0:
            super().__init__(*args, **kw)
        else:
            assert len(args) == 2
            assert args[0]._type_ is ct.c_char
            assert args[1]._length_ == 1
            super().__init__(*args, 1, **kw)


class SPARSE_MAT_PTR(ct.Structure):
    _fields_ = [
        ("values", DOUBLE_VEC_PTR),
        ("indices", LONG_VEC_PTR),
        ("offsets", LONG_VEC_PTR),
    ]


class SHAPE_PTR(ct.Structure):
    _fields_ = [
        ("type", ct.c_char_p),
        ("ischeme", ct.c_char_p),
    ]


class CONSTRAINTS_PTR(ct.Structure):
    _fields_ = [
        ("dofs", LONG_VEC_PTR),
        ("values", DOUBLE_VEC_PTR),
    ]


class POINTSET_PTR(ct.Structure):
    _fields_ = [
        ("data", DOUBLE_MAT_PTR),
    ]


class GROUPSET_PTR(ct.Structure):
    _fields_ = [
        ("data", LONG_MAT_PTR),
        ("sizes", LONG_VEC_PTR),
    ]


class DOFSPACE_PTR(ct.Structure):
    _fields_ = [
        ("data", LONG_MAT_PTR),
    ]


class GLOBDAT(ct.Structure):
    _fields_ = [
        ("nodeSet", POINTSET_PTR),
        ("elemSet", GROUPSET_PTR),
        ("dofSpace", DOFSPACE_PTR),
        ("state0", DOUBLE_VEC_PTR),
        ("intForce", DOUBLE_VEC_PTR),
        ("extForce", DOUBLE_VEC_PTR),
        ("matrix0", SPARSE_MAT_PTR),
        ("constraints", CONSTRAINTS_PTR),
        ("shape", SHAPE_PTR),
    ]


def to_numpy(c_obj, *args):
    if isinstance(c_obj, (LONG_ARRAY_PTR, DOUBLE_ARRAY_PTR)):
        assert len(args) == 0
        dim = c_obj.dim
        shape = tuple()
        for i in range(dim):
            shape += (c_obj.shape[i],)
        return np.ctypeslib.as_array(c_obj.ptr, shape).copy()

    elif isinstance(c_obj, POINTSET_PTR):
        assert len(args) == 0
        nodes = XNodeSet()
        nodes.add_nodes(to_numpy(c_obj.data))
        nodes.to_nodeset()
        return nodes

    elif isinstance(c_obj, GROUPSET_PTR):
        assert len(args) == 1
        nodes = args[0]
        elems = XElementSet(nodes)
        elems.add_elements(to_numpy(c_obj.data), to_numpy(c_obj.sizes))
        elems.to_elementset()
        return elems

    elif isinstance(c_obj, DOFSPACE_PTR):
        assert len(args) == 0
        dofs_data = to_numpy(c_obj.data)
        dofs = DofSpace()
        for it in range(dofs_data.shape[1]):
            dof_type = ["dx", "dy", "dz"][it]
            dofs.add_type(dof_type)

            for inode in range(dofs_data.shape[0]):
                dofs._dofs[dof_type][inode] = dofs_data[inode, it]

        dofs._count = dofs_data.shape[0] * dofs_data.shape[1]
        return dofs

    elif isinstance(c_obj, SPARSE_MAT_PTR):
        matrix0 = csr_array(
            (to_numpy(c_obj.values), to_numpy(c_obj.indices), to_numpy(c_obj.offsets))
        )
        return matrix0

    elif isinstance(c_obj, CONSTRAINTS_PTR):
        cons = Constraints()
        for dof, val in zip(to_numpy(c_obj.dofs), to_numpy(c_obj.values)):
            cons.add_constraint(dof, val)
        return cons

    elif isinstance(c_obj, SHAPE_PTR):
        dummy = {}
        declare_shapes(dummy)
        factory = dummy["shapeFactory"]
        shape = factory.get_shape(c_obj.type.decode(), c_obj.ischeme.decode())
        return shape

    else:
        assert False


def to_ctypes(py_obj):
    if py_obj is None:
        return None

    elif isinstance(py_obj, np.ndarray):
        ctype = np.ctypeslib.as_ctypes_type(py_obj.dtype)
        ptr = py_obj.ctypes.data_as(ct.POINTER(ctype))
        shape = py_obj.ctypes.shape_as(ct.c_long)

        if shape._length_ == 1:
            if ptr._type_ is ct.c_long:
                return LONG_VEC_PTR(ptr, shape)
            elif ptr._type_ is ct.c_double:
                return DOUBLE_VEC_PTR(ptr, shape)
            else:
                assert False
        elif shape._length_ == 2:
            if ptr._type_ is ct.c_long:
                return LONG_MAT_PTR(ptr, shape)
            elif ptr._type_ is ct.c_double:
                return DOUBLE_MAT_PTR(ptr, shape)
            else:
                assert False
        else:
            assert False

    elif isinstance(py_obj, NodeSet):
        return POINTSET_PTR(to_ctypes(py_obj.get_coords()))

    elif isinstance(py_obj, ElementSet):
        return GROUPSET_PTR(to_ctypes(py_obj._data), to_ctypes(py_obj._groupsizes))

    else:
        assert False


def initialize_globdat(*, node_count, elem_count, rank, max_elem_node_count):
    dof_count = node_count * rank

    nodes_data = np.zeros((node_count, rank), dtype=np.double)
    elems_data = np.zeros((elem_count, max_elem_node_count), dtype=np.long)
    elems_sizes = np.zeros(elem_count, dtype=np.long)
    dofs = np.zeros((node_count, rank), dtype=np.long)

    state0 = np.zeros(dof_count, dtype=np.double)
    intForce = np.zeros(dof_count, dtype=np.double)
    extForce = np.zeros(dof_count, dtype=np.double)

    matrix0_values = np.zeros(dof_count * 20, dtype=np.double)
    matrix0_indices = np.zeros(dof_count * 20, dtype=np.long)
    matrix0_offsets = np.zeros(dof_count + 1, dtype=np.long)

    constraints_dofs = np.zeros(dof_count, dtype=np.long)
    constraints_values = np.zeros(dof_count, dtype=np.double)

    globdat = {
        "nodeSet": {"data": nodes_data},
        "elemSet": {"data": elems_data, "sizes": elems_sizes},
        "dofSpace": {"data": dofs},
        "state0": state0,
        "intForce": intForce,
        "extForce": extForce,
        "matrix0": {
            "values": matrix0_values,
            "indices": matrix0_indices,
            "offsets": matrix0_offsets,
        },
        "constraints": {"dofs": constraints_dofs, "values": constraints_values},
        "shape": {"type": "", "ischeme": ""},
    }

    return globdat


def numpy_globdat_to_ctypes(np_globdat):
    if "nodeSet" in np_globdat:
        ct_nodes = to_ctypes(np_globdat.get("nodeSet"))
    else:
        ct_nodes = POINTSET_PTR()

    if "elemSet" in np_globdat:
        ct_elems = to_ctypes(np_globdat.get("elemSet"))
    else:
        ct_nodes = GROUPSET_PTR()

    if "dofSpace" in np_globdat:
        ct_dofs = DOFSPACE_PTR(to_ctypes(np_globdat["dofSpace"]["data"]))
    else:
        ct_dofs = DOFSPACE_PTR()

    if "state0" in np_globdat:
        ct_state0 = to_ctypes(np_globdat.get("state0"))
    else:
        ct_state0 = DOUBLE_VEC_PTR()

    if "intForce" in np_globdat:
        ct_intForce = to_ctypes(np_globdat.get("intForce"))
    else:
        ct_intForce = DOUBLE_VEC_PTR()

    if "extForce" in np_globdat:
        ct_extForce = to_ctypes(np_globdat.get("extForce"))
    else:
        ct_extForce = DOUBLE_VEC_PTR()

    if "matrix0" in np_globdat:
        ct_matrix0_values = to_ctypes(np_globdat["matrix0"]["values"])
        ct_matrix0_indices = to_ctypes(np_globdat["matrix0"]["indices"])
        ct_matrix0_offsets = to_ctypes(np_globdat["matrix0"]["offsets"])
        ct_matrix0 = SPARSE_MAT_PTR(
            ct_matrix0_values, ct_matrix0_indices, ct_matrix0_offsets
        )
    else:
        ct_matrix0 = SPARSE_MAT_PTR()

    if "constraints" in np_globdat:
        ct_constraints_dofs = to_ctypes(np_globdat["constraints"]["dofs"])
        ct_constraints_values = to_ctypes(np_globdat["constraints"]["values"])
        ct_constraints = CONSTRAINTS_PTR(ct_constraints_dofs, ct_constraints_values)
    else:
        ct_constraints = CONSTRAINTS_PTR()

    if "shape" in np_globdat:
        ct_shape_type = ct.create_string_buffer(
            np_globdat["shape"]["type"].encode(), 64
        )
        ct_shape_ischeme = ct.create_string_buffer(
            np_globdat["shape"]["ischeme"].encode(), 64
        )
        ct_shape = SHAPE_PTR(
            ct.cast(ct_shape_type, ct.c_char_p), ct.cast(ct_shape_ischeme, ct.c_char_p)
        )
    else:
        ct_shape = SHAPE_PTR()

    ct_globdat = GLOBDAT(
        ct_nodes,
        ct_elems,
        ct_dofs,
        ct_state0,
        ct_intForce,
        ct_extForce,
        ct_matrix0,
        ct_constraints,
        ct_shape,
    )

    return ct_globdat


def ctypes_globdat_to_numpy(ct_globdat):
    nodes = to_numpy(ct_globdat.nodeSet)
    elems = to_numpy(ct_globdat.elemSet, nodes)

    np_globdat = {
        "nodeSet": nodes,
        "elemSet": elems,
        "dofSpace": to_numpy(ct_globdat.dofSpace),
        "state0": to_numpy(ct_globdat.state0),
        "intForce": to_numpy(ct_globdat.intForce),
        "extForce": to_numpy(ct_globdat.extForce),
        "matrix0": to_numpy(ct_globdat.matrix0),
        "constraints": to_numpy(ct_globdat.constraints),
        "shape": to_numpy(ct_globdat.shape),
    }

    return np_globdat
