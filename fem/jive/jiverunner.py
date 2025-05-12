import os
import ctypes as ct
import numpy as np

from myjive.app import main
from myjive.util.proputils import props_to_string
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

    def __init__(
        self,
        props,
        *,
        elems=None,
        egroups=None,
        node_count=None,
        elem_count=None,
        rank=None,
        max_elem_node_count=None
    ):
        self.props = props

        if elems is None:
            assert egroups is None
            assert node_count is not None
            assert elem_count is not None
            assert rank is not None
            assert max_elem_node_count is not None

            self.elems = None
            self.egroups = None
            self.node_count = node_count
            self.elem_count = elem_count
            self.rank = rank
            self.max_elem_node_count = max_elem_node_count

        else:
            assert node_count is None
            assert elem_count is None
            assert rank is None
            assert max_elem_node_count is None

            self.update_elems(elems)

            if egroups is None:
                self.egroups = None
            else:
                self.egroups = egroups
                for egroup in self.egroups.values():
                    assert self.elems is egroup.get_elements()

    def __call__(self, *flags, **backdoor):
        flags = list(flags)

        if len(flags) == 0:
            flags = [
                "nodeSet",
                "elementSet",
                "elementGroups",
                "dofSpace",
                "state0",
                "extForce",
                "intForce",
                "matrix0",
                "constraints",
                "shape",
            ]
        else:
            if self.elems is not None:
                if "elementSet" not in flags:
                    flags.append("elementSet")
                if "nodeSet" not in flags:
                    flags.append("nodeSet")
            if self.egroups is not None:
                if "elementGroups" not in flags:
                    flags.append("elementGroups")

        buffers = ctutil.initialize_buffers(
            node_count=self.node_count,
            elem_count=self.elem_count,
            ngroup_count=0,
            egroup_count=0 if self.egroups is None else len(self.egroups),
            rank=self.rank,
            max_elem_node_count=self.max_elem_node_count,
            flags=flags,
        )

        if self.elems is not None:
            assert "elementSet" in buffers
            buffers["elementSet"] = ctutil.to_buffer(self.elems)
            assert "nodeSet" in buffers
            buffers["nodeSet"] = ctutil.to_buffer(self.elems.get_nodes())

        if self.egroups is not None:
            assert "elementGroups" in buffers
            buffers["elementGroups"] = ctutil.to_buffer(self.egroups)

        nbac = len(backdoor)
        if nbac > 0:
            buffers["backdoor"] = {}
            fields = np.array(list(backdoor.keys()))
            values = np.array(list(backdoor.values())).reshape((nbac, -1)).copy()

            buffers["backdoor"]["ipfields"] = fields
            buffers["backdoor"]["ipvalues"] = values

        ct_globdat = ctutil.buffers_as_ctypes(buffers)
        ct_flags = ct.c_long(ctutil.pack_output_flags(*flags))

        loader = ct.LibraryLoader(ct.CDLL)
        abspath = os.path.abspath(os.path.join(__file__, "..", "src", "liblinear.so"))
        liblinear = loader.LoadLibrary(abspath)

        assert isinstance(self.props, (str, dict))

        if isinstance(self.props, dict):
            # pass the props directly to jive
            str_props = props_to_string(self.props).encode()

            runFromProps = liblinear.runFromProps
            runFromProps.argtypes = (
                ct.POINTER(ctutil.GLOBDAT),
                ct.POINTER(ct.c_char),
                ct.c_long,
            )
            runFromProps(ct.byref(ct_globdat), str_props, ct_flags)

        else:
            # get the props from a file
            fname = self.props.encode()

            runFromFile = liblinear.runFromFile
            runFromFile.argtypes = (
                ct.POINTER(ctutil.GLOBDAT),
                ct.POINTER(ct.c_char),
                ct.c_long,
            )
            runFromFile(ct.byref(ct_globdat), fname, ct_flags)

        np_globdat = ctutil.ctypes_globdat_to_numpy(ct_globdat, flags)

        return np_globdat

    def update_elems(self, elems):
        self.elems = elems
        nodes = self.elems.get_nodes()
        self.node_count = len(nodes)
        self.elem_count = len(elems)
        self.rank = nodes.rank()
        self.max_elem_node_count = elems.max_elem_node_count()
