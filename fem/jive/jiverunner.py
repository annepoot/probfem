import os
import numpy as np
import ctypes as ct

from myjive.app import main
from myjivex.declare import declare_all

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

        state0_func = liblinear.getState0
        state0_func.argtypes = (
            PTR(ct.c_double),
            PTR(ct.c_int),
            PTR(ct.c_double),
            PTR(ct.c_int),
            PTR(ct.c_int),
            PTR(ct.c_int),
            PTR(ct.c_int),
            PTR(ct.c_int),
            PTR(ct.c_char),
        )

        dof_count = self.node_count * self.rank

        state0_ptr = (ct.c_double * dof_count)()
        state0_size = ct.c_int(dof_count)
        coords_ptr = (ct.c_double * (self.node_count * self.rank))()
        coords_size = ct.c_int(self.node_count)
        coords_rank = ct.c_int(self.rank)
        dofs_ptr = (ct.c_int * (self.node_count * self.rank))()
        dofs_size = ct.c_int(self.node_count)
        dofs_rank = ct.c_int(self.rank)

        state0_func(
            state0_ptr,
            ct.byref(state0_size),
            coords_ptr,
            ct.byref(coords_size),
            ct.byref(coords_rank),
            dofs_ptr,
            ct.byref(dofs_size),
            ct.byref(dofs_rank),
            self.fname,
        )

        state0 = np.resize(np.ctypeslib.as_array(state0_ptr), state0_size.value)
        coords = np.resize(
            np.ctypeslib.as_array(coords_ptr), (coords_size.value, coords_rank.value)
        )
        dof_idx = np.resize(
            np.ctypeslib.as_array(dofs_ptr), (dofs_size.value, dofs_rank.value)
        )

        globdat = {"state0": state0, "coords": coords, "dofs": dof_idx}

        return globdat
