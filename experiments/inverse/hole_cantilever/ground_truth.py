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

runner = CJiveRunner(
    "props/fem.pro", node_count=45406, elem_count=89668, rank=2, max_elem_node_count=3
)

globdat = runner(input_globdat={"nodeSet": nodes, "elementSet": elems})

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

obs_inodes = np.arange(2, len(obs_locsx) + 2)

for inode, locx, locy in zip(obs_inodes, obs_locsx, obs_locsy):
    tol = 1e-8
    coord = np.array([locx, locy])
    assert np.allclose(nodes[inode], coord)

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
