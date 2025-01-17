import numpy as np
import os
import ctypes as ct
import time

from myjive.app import main
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

        np_globdat = ctutil.initialize_globdat(
            node_count=self.node_count,
            elem_count=self.elem_count,
            rank=self.rank,
            max_elem_node_count=self.max_elem_node_count,
        )

        for key, val in input_globdat.items():
            assert key in np_globdat
            np_globdat[key] = val

        ct_globdat = ctutil.numpy_globdat_to_ctypes(np_globdat)
        globdat_func(ct.byref(ct_globdat), fname)
        np_globdat = ctutil.ctypes_globdat_to_numpy(ct_globdat)

        if tmp_file:
            os.remove(fname)

        return np_globdat
