import numpy as np
import ctypes as ct
from scipy.sparse import csr_array

from myjive.fem import NodeSet, XNodeSet, ElementSet, XElementSet, DofSpace
from myjive.names import GlobNames as gn
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
        ("elementSet", GROUPSET_PTR),
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


def to_ctypes(arr):
    ctype = np.ctypeslib.as_ctypes_type(arr.dtype)
    ptr = arr.ctypes.data_as(ct.POINTER(ctype))
    shape = arr.ctypes.shape_as(ct.c_long)

    assert arr.base is None

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


def to_buffer(py_obj):
    if isinstance(py_obj, NodeSet):
        size = len(py_obj)
        return {"data": py_obj._data[:size].copy()}
    elif isinstance(py_obj, ElementSet):
        size = len(py_obj)
        return {
            "data": py_obj._data[:size].copy(),
            "sizes": py_obj._groupsizes[:size].copy(),
        }
    else:
        assert False


def initialize_buffers(*, node_count, elem_count, rank, max_elem_node_count, flags):
    dof_count = node_count * rank

    if len(flags) == 0:
        nodes_data = np.zeros((node_count, rank), dtype=np.double)
        elems_data = np.zeros((elem_count, max_elem_node_count), dtype=int)
        elems_sizes = np.zeros(elem_count, dtype=int)
        dofs = np.zeros((node_count, rank), dtype=int)

        state0 = np.zeros(dof_count, dtype=np.double)
        intForce = np.zeros(dof_count, dtype=np.double)
        extForce = np.zeros(dof_count, dtype=np.double)

        matrix0_values = np.zeros(dof_count * 20, dtype=np.double)
        matrix0_indices = np.zeros(dof_count * 20, dtype=int)
        matrix0_offsets = np.zeros(dof_count + 1, dtype=int)

        constraints_dofs = np.zeros(dof_count, dtype=int)
        constraints_values = np.zeros(dof_count, dtype=np.double)

        shape_type = ct.cast(ct.create_string_buffer(64), ct.c_char_p)
        shape_ischeme = ct.cast(ct.create_string_buffer(64), ct.c_char_p)

        buffers = {
            "nodeSet": {"data": nodes_data},
            "elementSet": {"data": elems_data, "sizes": elems_sizes},
            "dofSpace": {"data": dofs},
            "state0": state0,
            "extForce": extForce,
            "intForce": intForce,
            "matrix0": {
                "values": matrix0_values,
                "indices": matrix0_indices,
                "offsets": matrix0_offsets,
            },
            "constraints": {"dofs": constraints_dofs, "values": constraints_values},
            "shape": {"type": shape_type, "ischeme": shape_ischeme},
        }

    else:
        buffers = {}

        for flag in flags:
            if flag == "nodeSet":
                nodes_data = np.zeros((node_count, rank), dtype=np.double)
                buffers[flag] = {"data": nodes_data}

            elif flag == "elementSet":
                elems_data = np.zeros((elem_count, max_elem_node_count), dtype=int)
                elems_sizes = np.zeros(elem_count, dtype=int)
                buffers[flag] = {"data": elems_data, "sizes": elems_sizes}

            elif flag == "dofSpace":
                dofs = np.zeros((node_count, rank), dtype=int)
                buffers[flag] = {"data": dofs}

            elif flag == "state0":
                state0 = np.zeros(dof_count, dtype=np.double)
                buffers[flag] = state0

            elif flag == "extForce":
                extForce = np.zeros(dof_count, dtype=np.double)
                buffers[flag] = extForce

            elif flag == "intForce":
                intForce = np.zeros(dof_count, dtype=np.double)
                buffers[flag] = intForce

            elif flag == "matrix0":
                matrix0_values = np.zeros(dof_count * 20, dtype=np.double)
                matrix0_indices = np.zeros(dof_count * 20, dtype=int)
                matrix0_offsets = np.zeros(dof_count + 1, dtype=int)
                buffers[flag] = {
                    "values": matrix0_values,
                    "indices": matrix0_indices,
                    "offsets": matrix0_offsets,
                }

            elif flag == "constraints":
                constraints_dofs = np.zeros(dof_count, dtype=int)
                constraints_values = np.zeros(dof_count, dtype=np.double)
                buffers[flag] = {"dofs": constraints_dofs, "values": constraints_values}

            elif flag == "shape":
                shape_type = ct.cast(ct.create_string_buffer(64), ct.c_char_p)
                shape_ischeme = ct.cast(ct.create_string_buffer(64), ct.c_char_p)
                buffers[flag] = {"type": shape_type, "ischeme": shape_ischeme}

            else:
                assert False

    return buffers


def buffers_as_ctypes(buffers):

    if "nodeSet" in buffers:
        nodeset_ptr = POINTSET_PTR(to_ctypes(buffers["nodeSet"]["data"]))
    else:
        nodeset_ptr = POINTSET_PTR()

    if "elementSet" in buffers:
        elementset_ptr = GROUPSET_PTR(
            to_ctypes(buffers["elementSet"]["data"]),
            to_ctypes(buffers["elementSet"]["sizes"]),
        )
    else:
        elementset_ptr = GROUPSET_PTR()

    if "dofSpace" in buffers:
        dofspace_ptr = DOFSPACE_PTR(to_ctypes(buffers["dofSpace"]["data"]))
    else:
        dofspace_ptr = DOFSPACE_PTR()

    if "state0" in buffers:
        state0_ptr = to_ctypes(buffers["state0"])
    else:
        state0_ptr = DOUBLE_VEC_PTR()

    if "extForce" in buffers:
        extforce_ptr = to_ctypes(buffers["extForce"])
    else:
        extforce_ptr = DOUBLE_VEC_PTR()

    if "intForce" in buffers:
        intforce_ptr = to_ctypes(buffers["intForce"])
    else:
        intforce_ptr = DOUBLE_VEC_PTR()

    if "matrix0" in buffers:
        matrix0_ptr = SPARSE_MAT_PTR(
            to_ctypes(buffers["matrix0"]["values"]),
            to_ctypes(buffers["matrix0"]["indices"]),
            to_ctypes(buffers["matrix0"]["offsets"]),
        )
    else:
        matrix0_ptr = SPARSE_MAT_PTR()

    if "constraints" in buffers:
        constraints_ptr = CONSTRAINTS_PTR(
            to_ctypes(buffers["constraints"]["dofs"]),
            to_ctypes(buffers["constraints"]["values"]),
        )
    else:
        constraints_ptr = CONSTRAINTS_PTR()

    if "shape" in buffers:
        shape_ptr = SHAPE_PTR(buffers["shape"]["type"], buffers["shape"]["ischeme"])
    else:
        shape_ptr = SHAPE_PTR()

    ct_globdat = GLOBDAT(
        nodeset_ptr,
        elementset_ptr,
        dofspace_ptr,
        state0_ptr,
        extforce_ptr,
        intforce_ptr,
        matrix0_ptr,
        constraints_ptr,
        shape_ptr,
    )

    return ct_globdat


def ctypes_globdat_to_numpy(ct_globdat, flags):
    if len(flags) == 0:
        nodes = to_numpy(ct_globdat.nodeSet)
        elems = to_numpy(ct_globdat.elementSet, nodes)

        np_globdat = {
            gn.NSET: nodes,
            gn.ESET: elems,
            gn.DOFSPACE: to_numpy(ct_globdat.dofSpace),
            gn.STATE0: to_numpy(ct_globdat.state0),
            gn.EXTFORCE: to_numpy(ct_globdat.extForce),
            gn.INTFORCE: to_numpy(ct_globdat.intForce),
            gn.MATRIX0: to_numpy(ct_globdat.matrix0),
            gn.CONSTRAINTS: to_numpy(ct_globdat.constraints),
            gn.SHAPE: to_numpy(ct_globdat.shape),
        }

    else:
        if "nodeSet" in flags or "elementSet" in flags:
            nodes = to_numpy(ct_globdat.nodeSet)
            elems = to_numpy(ct_globdat.elementSet, nodes)

        np_globdat = {}

        for flag in flags:
            if flag == "nodeSet":
                np_globdat[gn.NSET] = nodes

            elif flag == "elementSet":
                np_globdat[gn.ESET] = elems

            elif flag == "dofSpace":
                np_globdat[gn.DOFSPACE] = to_numpy(ct_globdat.dofSpace)

            elif flag == "state0":
                np_globdat[gn.STATE0] = to_numpy(ct_globdat.state0)

            elif flag == "extForce":
                np_globdat[gn.EXTFORCE] = to_numpy(ct_globdat.extForce)

            elif flag == "intForce":
                np_globdat[gn.INTFORCE] = to_numpy(ct_globdat.intForce)

            elif flag == "matrix0":
                np_globdat[gn.MATRIX0] = to_numpy(ct_globdat.matrix0)

            elif flag == "constraints":
                np_globdat[gn.CONSTRAINTS] = to_numpy(ct_globdat.constraints)

            elif flag == "shape":
                np_globdat[gn.SHAPE] = to_numpy(ct_globdat.shape)

            else:
                assert False

    return np_globdat


flag_map = {
    "nodeSet": 1 << 0,
    "elementSet": 1 << 1,
    "dofSpace": 1 << 2,
    "state0": 1 << 3,
    "intForce": 1 << 4,
    "extForce": 1 << 5,
    "matrix0": 1 << 6,
    "constraints": 1 << 7,
    "shape": 1 << 8,
}


def pack_output_flags(*flags):
    pack = 0
    for flag in flags:
        shifted_bit = flag_map.get(flag)
        assert shifted_bit is not None
        pack += shifted_bit
    return pack


def unpack_output_flags(pack):
    flags = []
    for flag, shifted_bit in flag_map.items():
        if pack & shifted_bit > 0:
            flags.append(flag)
    return flags
