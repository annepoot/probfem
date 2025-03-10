import numpy as np
import pandas as pd
from fem.jive import CJiveRunner
from fem.meshing import find_coords_in_nodeset

from experiments.inverse.three_point_hole.meshing import (
    create_mesh,
    get_observation_locations,
)
from experiments.inverse.three_point_hole.props import get_fem_props

h = 0.002
L = 5.0
H = 1.0
U = 0.5
x = 1.0
y = 0.4
a = 0.4
theta = np.pi / 6
r_rel = 0.25
h_meas = 0.1

nodes, elems = create_mesh(
    h=h,
    L=L,
    H=H,
    U=U,
    x=x,
    y=y,
    a=a,
    theta=theta,
    r_rel=r_rel,
    h_meas=h_meas,
)

jive = CJiveRunner(props=get_fem_props(), elems=elems)
globdat = jive()

state0 = globdat["state0"]
coords = globdat["nodeSet"].get_coords()
dofs = globdat["dofSpace"]
types = np.array(["dx", "dy"])

obs_locs = get_observation_locations(L=L, H=H, h_meas=h_meas)
obs_inodes = find_coords_in_nodeset(obs_locs, nodes)

obs_dofs = np.tile(np.arange(0, 2), len(obs_inodes))
obs_locsx = np.repeat(obs_locs[:, 0], 2)
obs_locsy = np.repeat(obs_locs[:, 1], 2)
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
