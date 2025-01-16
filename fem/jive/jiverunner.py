import os
import ctypes as ct
from scipy.sparse import csr_array
import time

from myjive.app import main
from myjive.declare import declare_shapes
from myjive.fem import XNodeSet, XElementSet, DofSpace
from myjive.solver import Constraints
from myjive.util.proputils import write_to_file
from myjivex.declare import declare_all

from fem.jive import ctypesutils as ctutil

__all__ = ["MyJiveRunner", "CJiveRunner"]


class MyJiveRunner:

    def __init__(self, props):
        self.props = props

    def __call__(self):
        globdat = main.jive(self.props, extra_declares=[declare_all])
        return globdat


class CJiveRunner:

    def __init__(self, props, *, node_count, elem_count, rank, max_elem_node_count):
        self.props = props
        self.node_count = node_count
        self.elem_count = elem_count
        self.rank = rank
        self.max_elem_node_count = max_elem_node_count

    def __call__(self, input_globdat={}):

        loader = ct.LibraryLoader(ct.CDLL)
        abspath = os.path.abspath(os.path.join(__file__, "..", "src", "liblinear.so"))
        liblinear = loader.LoadLibrary(abspath)

        globdat_func = liblinear.getGlobdat
        globdat_func.argtypes = (
            ct.POINTER(ctutil.GLOBDAT),
            ct.POINTER(ct.c_char),
        )

        assert isinstance(self.props, (str, dict))
        tmp_file = isinstance(self.props, dict)

        if tmp_file:
            fname = "tmp" + time.strftime("%Y%m%d%H%M%S") + ".pro"
            write_to_file(self.props, fname)
        else:
            fname = self.props
        fname = fname.encode("utf-8")

        globdat = ctutil.initialize_globdat(
            node_count=self.node_count,
            elem_count=self.elem_count,
            rank=self.rank,
            max_elem_node_count=self.max_elem_node_count,
        )

        globdat_func(
            ct.byref(globdat),
            fname,
        )

        if tmp_file:
            os.remove(fname)

        state0 = ctutil.to_numpy(globdat.state0)
        intForce = ctutil.to_numpy(globdat.intForce)
        extForce = ctutil.to_numpy(globdat.extForce)
        dofs_data = ctutil.to_numpy(globdat.dofSpace)

        nodes_data = ctutil.to_numpy(globdat.nodeSet.data)
        elems_data = ctutil.to_numpy(globdat.elemSet.data)
        elems_sizes = ctutil.to_numpy(globdat.elemSet.sizes)

        matrix0_values = ctutil.to_numpy(globdat.matrix0.values)
        matrix0_indices = ctutil.to_numpy(globdat.matrix0.indices)
        matrix0_offsets = ctutil.to_numpy(globdat.matrix0.offsets)

        matrix0 = csr_array((matrix0_values, matrix0_indices, matrix0_offsets))

        cdofs = ctutil.to_numpy(globdat.constraints.dofs)
        cvals = ctutil.to_numpy(globdat.constraints.values)

        constraints = Constraints()
        for cdof, cval in zip(cdofs, cvals):
            constraints.add_constraint(cdof, cval)

        nodes = XNodeSet()
        nodes.add_nodes(nodes_data)
        nodes.to_nodeset()

        elems = XElementSet(nodes)
        elems.add_elements(elems_data, elems_sizes)
        elems.to_elementset

        dofs = DofSpace()
        for it in range(dofs_data.shape[1]):
            dof_type = ["dx", "dy", "dz"][it]
            dofs.add_type(dof_type)

            for inode in range(dofs_data.shape[0]):
                dofs._dofs[dof_type][inode] = dofs_data[inode, it]

        dofs._count = dofs_data.shape[0] * dofs_data.shape[1]

        shape = globdat.shape
        shape_type = shape.type.ptr[: shape.type.size].decode("UTF-8")
        shape_ischeme = shape.ischeme.ptr[: shape.ischeme.size].decode("UTF-8")

        globdat_ = {
            "nodeSet": nodes,
            "elemSet": elems,
            "dofSpace": dofs,
            "state0": state0,
            "intForce": intForce,
            "extForce": extForce,
            "matrix0": matrix0,
            "constraints": constraints,
            "shape": {"type": shape_type, "ischeme": shape_ischeme},
        }
        declare_shapes(globdat_)
        factory = globdat_["shapeFactory"]
        globdat_["shape"] = factory.get_shape(shape_type, shape_ischeme)

        return globdat_
