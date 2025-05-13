import os
import numpy as np

from fem.jive import CJiveRunner
from fem.meshing import read_mesh
from experiments.inverse.frp_damage.props import get_fem_props
from experiments.inverse.frp_damage import caching, misc, params


# Reference values for material properties from
# On the response to hygrothermal aging of pultruded FRPs used in the civil engineering sector
# by Grammatikos et al. (2016), table 2
# https://www.sciencedirect.com/science/article/pii/S0264127516301733

props = get_fem_props()
props["model"]["model"]["models"].append("neum")
props["model"]["model"]["diri"] = {
    "type": "Dirichlet",
    "initDisp": 0.0,
    "dispIncr": 0.0,
    "nodeGroups": ["b", "b"],
    "dofs": ["dx", "dy"],
    "factors": [0.0, 0.0],
}
props["model"]["model"]["neum"] = {
    "type": "Neumann",
    "initLoad": 1.0,
    "loadIncr": 0.0,
    "nodeGroups": ["t"],
    "dofs": ["dx"],
    "factors": [1.0],
}

E_fiber = 22000.0  # 22 GPa (unaged, tensile stiffness)
nu_fiber = 0.2
G_matrix = 3800.0  # 3.8 GPa (unaged, in-plane shear stiffness)
nu_matrix = 0.2
E_matrix = 9120.0  # 2 * G_matrix * (1 + nu_matrix)

# conclusion: the following parameter setting is all right:
# E = 9120
# alpha = 50.0
# beta = 25.0
# c = 0.02
# d = 0.5
# This results in a 15% decrease in shear stiffness.
# max horizontal displacement goes from 0.15 to 0.17 (h=0.01, nfib=30)

props["model"]["model"]["fiber"]["material"]["E"] = E_fiber
props["model"]["model"]["fiber"]["material"]["nu"] = nu_fiber
props["model"]["model"]["matrix"]["material"]["E"] = "backdoor"
props["model"]["model"]["matrix"]["material"]["nu"] = nu_matrix

n_fiber = params.geometry_params["n_fiber"]
r_fiber = params.geometry_params["r_fiber"]
h = 0.01

fname = "meshes/rve_h-{:.3f}_nfib-{}.msh".format(h, n_fiber)
nodes, elems, groups = read_mesh(fname, read_groups=True)

from myjive.fem import Tri3Shape

shape = Tri3Shape("Gauss3")
egroup = groups["matrix"]
ipoints = caching.get_or_calc_ipoints(egroup=egroup, h=h)
distances = caching.get_or_calc_distances(egroup=egroup, h=h)

backdoor = {}
backdoor["xcoord"] = ipoints[:, 0]
backdoor["ycoord"] = ipoints[:, 1]
backdoor["e"] = np.zeros(ipoints.shape[0])

E = 9120
alpha = 50.0
beta = 25.0
c = 0.02
d = 0.5

for ip, ipoint in enumerate(ipoints):
    dist = distances[ip]
    sat = misc.saturation(dist, alpha, beta, c)
    dam = misc.damage(sat, d)
    backdoor["e"][ip] = E * (1 - dam)

jive = CJiveRunner(props, elems=elems, egroups=groups)
globdat = jive(**backdoor)

from myjivex.util import QuickViewer

QuickViewer(
    globdat["state0"],
    globdat,
    comp=0,
)
