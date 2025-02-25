import os
import ctypes as ct

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
        node_count=None,
        elem_count=None,
        rank=None,
        max_elem_node_count=None
    ):
        self.props = props

        if elems is None:
            assert node_count is not None
            assert elem_count is not None
            assert rank is not None
            assert max_elem_node_count is not None

            self.elems = None
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

    def __call__(self):

        buffers = ctutil.initialize_buffers(
            node_count=self.node_count,
            elem_count=self.elem_count,
            rank=self.rank,
            max_elem_node_count=self.max_elem_node_count,
        )

        if self.elems is not None:
            assert "elementSet" in buffers
            buffers["elementSet"] = ctutil.to_buffer(self.elems)
            assert "nodeSet" in buffers
            buffers["nodeSet"] = ctutil.to_buffer(self.elems.get_nodes())

        ct_globdat = ctutil.buffers_as_ctypes(buffers)

        loader = ct.LibraryLoader(ct.CDLL)
        abspath = os.path.abspath(os.path.join(__file__, "..", "src", "liblinear.so"))
        liblinear = loader.LoadLibrary(abspath)

        assert isinstance(self.props, (str, dict))

        if isinstance(self.props, dict):
            # pass the props directly to jive
            str_props = props_to_string(self.props).encode()

            runFromProps = liblinear.runFromProps
            runFromProps.argtypes = (ct.POINTER(ctutil.GLOBDAT), ct.POINTER(ct.c_char))
            runFromProps(ct.byref(ct_globdat), str_props)

        else:
            # get the props from a file
            fname = self.props.encode()

            runFromFile = liblinear.runFromFile
            runFromFile.argtypes = (ct.POINTER(ctutil.GLOBDAT), ct.POINTER(ct.c_char))
            runFromFile(ct.byref(ct_globdat), fname)

        np_globdat = ctutil.ctypes_globdat_to_numpy(ct_globdat)

        return np_globdat

    def update_elems(self, elems):
        self.elems = elems
        nodes = self.elems.get_nodes()
        self.node_count = len(nodes)
        self.elem_count = len(elems)
        self.rank = nodes.rank()
        self.max_elem_node_count = elems.max_elem_node_count()
