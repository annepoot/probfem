import os
import ctypes as ct
import numpy as np
import pandas as pd
from cantilever_mesh import create_mesh

PTR = ct.POINTER
ptr = ct.pointer

loader = ct.LibraryLoader(ct.CDLL)
abspath = os.path.abspath(os.path.join("..", "src", "liblinear.so"))
liblinear = loader.LoadLibrary(abspath)

state0_func = liblinear.getState0
state0_func.argtypes = (
    PTR(PTR(ct.c_double)),
    PTR(ct.c_int),
    PTR(PTR(ct.c_double)),
    PTR(ct.c_int),
    PTR(ct.c_int),
    PTR(PTR(ct.c_int)),
    PTR(ct.c_int),
    PTR(ct.c_int),
    PTR(ct.c_char),
)

state0_ptr = ptr(ct.c_double())
state0_size = ct.c_int()
coords_ptr = ptr(ct.c_double())
coords_rank = ct.c_int()
coords_size = ct.c_int()
dofs_ptr = ptr(ct.c_int())
dofs_rank = ct.c_int()
dofs_size = ct.c_int()

lc = 0.01
L = 4
H = 1
x = 1
y = 0.4
a = 0.4
theta = np.pi / 6
r = 0.1

create_mesh(lc=lc, L=L, H=H, x=x, y=y, a=a, theta=theta, r=r)

state0_func(
    ct.byref(state0_ptr),
    ct.byref(state0_size),
    ct.byref(coords_ptr),
    ct.byref(coords_rank),
    ct.byref(coords_size),
    ct.byref(dofs_ptr),
    ct.byref(dofs_rank),
    ct.byref(dofs_size),
    b"fem_props.pro",
)

state0 = np.ctypeslib.as_array(state0_ptr, (state0_size.value,)).copy()
coords = np.ctypeslib.as_array(
    coords_ptr, (coords_size.value, coords_rank.value)
).copy()
dof_idx = np.ctypeslib.as_array(dofs_ptr, (dofs_size.value, dofs_rank.value)).copy()

from myjive.fem import DofSpace
from fem.meshing import read_mesh

dofs = DofSpace()

nodes, elems = read_mesh("cantilever.msh")
types = np.array(["dx", "dy"])
for doftype in types:
    dofs.add_type(doftype)

for idof in range(dof_idx.shape[0] * dof_idx.shape[1]):
    inodes, itypes = np.where(dof_idx == idof)
    assert len(inodes) == len(itypes) == 1
    dofs.add_dof(inodes[0], types[itypes[0]])

obs_locsx = np.repeat(np.linspace(1, 4, 4), 4)
obs_locsy = np.tile(np.repeat(np.linspace(0, 1, 2), 2), 4)
obs_inodes = np.zeros(16, dtype=int)
for i, (locx, locy) in enumerate(zip(obs_locsx, obs_locsy)):
    tol = 1e-8
    coord = np.array([locx, locy])
    inodes = np.where(np.all(abs(coords - coord) < tol, axis=1))[0]
    assert len(inodes) == 1
    obs_inodes[i] = inodes[0]

obs_dofs = np.tile(np.arange(0, 2), 8)
obs_tdofs = types[obs_dofs]
obs_idofs = dof_idx[obs_inodes, obs_dofs]
obs_vals = state0[obs_idofs]

df = pd.DataFrame(
    {
        "loc_x": obs_locsx,
        "loc_y": obs_locsy,
        "inode": obs_inodes,
        "dof_idx": obs_dofs,
        "dof_type": obs_tdofs,
        "idof": obs_idofs,
        "value": obs_vals,
    }
)

df.to_csv("ground-truth.csv", index=False, header=True)
