import os
import numpy as np

from fem.jive import CJiveRunner
from fem.meshing import read_mesh
from experiments.inverse.frp_damage.props import get_fem_props
from experiments.inverse.frp_damage.meshing import calc_closest_fiber
from experiments.inverse.frp_damage import caching

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

E_fiber = 22000  # 22 GPa (unaged, tensile stiffness)
nu_fiber = 0.2
G_matrix = 3800  # 3.8 GPa (unaged, in-plane shear stiffness)
nu_matrix = 0.2
E_matrix = 9120  # 2 * G_matrix * (1 + nu_matrix)

# conclusion: reduction = 0.8 and decay = 20 is all right
# results in a 11% decrease in shear stiffness.
# max horizontal displacement goes from 0.09 to 0.10 (h=0.01, nfib=30)

props["model"]["model"]["fiber"]["E"] = E_fiber
props["model"]["model"]["fiber"]["nu"] = nu_fiber
props["model"]["model"]["matrix"]["E"] = E_matrix
props["model"]["model"]["matrix"]["nu"] = nu_matrix

fname = props["userinput"]["gmsh"]["file"]

base = os.path.splitext(os.path.basename(fname))[0]
for keyval in base.split("_"):
    if "-" in keyval:
        key, val = keyval.split("-")
        if key == "nfib":
            n_fiber = int(val)
        elif key == "h":
            h = float(val)

nodes, elems, groups = read_mesh(fname, read_groups=True)

from myjive.fem import Tri3Shape

shape = Tri3Shape("Gauss3")
egroup = groups["matrix"]

name = "fibers"
dependencies = {"nfib": n_fiber}
path = caching.get_cache_fpath(name, dependencies)
fibers = caching.read_cache(path)

name = "ipoints"
dependencies = {"nfib": n_fiber, "h": h}
path = caching.get_cache_fpath(name, dependencies)
ipoints = caching.read_cache(path)

name = "distances"
dependencies = {"nfib": n_fiber, "h": h}
path = caching.get_cache_fpath(name, dependencies)
distances = caching.read_cache(path)

backdoor = {}
backdoor["xcoord"] = ipoints[:, 0]
backdoor["ycoord"] = ipoints[:, 1]
backdoor["e"] = np.zeros(ipoints.shape[0])

decay = 20
reduction = 0.8

for ip, ipoint in enumerate(ipoints):
    dist = distances[ip]
    assert dist >= 0.09
    surf_dist = max(dist - 0.1, 0.0)
    moisture = np.exp(-decay * surf_dist)
    damage = reduction * moisture
    backdoor["e"][ip] = E_matrix * (1 - damage)

jive = CJiveRunner(props, elems=elems)
globdat = jive(**backdoor)

from myjivex.util import QuickViewer

QuickViewer(
    globdat["state0"],
    globdat,
    comp=0,
    title=r"max strain, $N_e = {}$".format(len(elems)),
)
