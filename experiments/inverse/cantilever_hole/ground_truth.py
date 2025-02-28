import numpy as np
import pandas as pd
from fem.jive import CJiveRunner

from experiments.inverse.cantilever_hole.meshing import create_mesh
from experiments.inverse.cantilever_hole.props import get_fem_props

h = 0.002
L = 4.0
H = 1.0
x = 1.0
y = 0.4
a = 0.4
theta = np.pi / 6
r_rel = 0.25
h_meas = 1.0

nodes, elems = create_mesh(
    h=h,
    L=L,
    H=H,
    x=x,
    y=y,
    a=a,
    theta=theta,
    r_rel=r_rel,
    h_meas=h_meas,
)[0]

jive = CJiveRunner(props=get_fem_props(), elems=elems)
globdat = jive()

state0 = globdat["state0"]
coords = globdat["nodeSet"].get_coords()
dofs = globdat["dofSpace"]

types = np.array(["dx", "dy"])

obs_locsx = []
obs_locsy = []
for x_point in np.linspace(0, L, int(L / h_meas) + 1)[1:]:
    for y_point in [0.0, H]:
        obs_locsx.append(x_point)
        obs_locsy.append(y_point)

for y_point in np.linspace(0, H, int(H / h_meas) + 1)[1:-1]:
    x_point = L
    obs_locsx.append(x_point)
    obs_locsy.append(y_point)

obs_inodes = []
for i, (locx, locy) in enumerate(zip(obs_locsx, obs_locsy)):
    loc = np.array([locx, locy])
    inodes = np.where(np.all(abs(coords - loc) < 1e-8, axis=1))[0]
    assert len(inodes) == 1
    obs_inodes.append(inodes[0])

obs_dofs = np.tile(np.arange(0, 2), len(obs_inodes))
obs_locsx = np.repeat(obs_locsx, 2)
obs_locsy = np.repeat(obs_locsy, 2)
obs_inodes = np.repeat(obs_inodes, 2)

obs_tdofs = types[obs_dofs]
obs_idofs = [dofs.get_dof(inode, dof) for inode, dof in zip(obs_inodes, obs_tdofs)]
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
