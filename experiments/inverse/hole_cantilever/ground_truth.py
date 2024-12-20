import numpy as np
import pandas as pd
from fem.jive import CJiveRunner

from experiments.inverse.hole_cantilever.meshing import create_mesh

h = 0.01
L = 4
H = 1
x = 1
y = 0.4
a = 0.4
theta = np.pi / 6
r_rel = 0.25

create_mesh(
    h=h,
    L=L,
    H=H,
    x=x,
    y=y,
    a=a,
    theta=theta,
    r_rel=r_rel,
    fname="cantilever.msh",
)

runner = CJiveRunner("props/fem.pro", node_count=45406, rank=2)
globdat = runner()

state0 = globdat["state0"]
coords = globdat["coords"]
dof_idx = globdat["dofs"]

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

create_mesh(
    h=0.1,
    L=L,
    H=H,
    x=x,
    y=y,
    a=a,
    theta=theta,
    r_rel=r_rel,
    fname="cantilever.msh",
)
