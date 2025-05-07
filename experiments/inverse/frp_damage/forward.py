import os
import numpy as np

from fem.jive import CJiveRunner
from fem.meshing import read_mesh
from experiments.inverse.frp_damage.props import get_fem_props
from experiments.inverse.frp_damage.meshing import calc_closest_fiber
from experiments.inverse.frp_damage import caching

props = get_fem_props()
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


def calc_integration_points(egroup, shape):
    elems = egroup.get_elements()
    nodes = elems.get_nodes()

    ipcount = len(egroup) * shape.ipoint_count()
    ipoints = np.zeros((ipcount, nodes.rank()))
    ip = 0

    for ielem in egroup:
        inodes = elems[ielem]
        coords = nodes[inodes]

        for ipoint in shape.get_global_integration_points(coords):
            ipoints[ip] = ipoint
            ip += 1

    if ip != ipcount:
        raise RuntimeError("Mismatched number of integration points")

    return ipoints


name = "fibers"
dependencies = {"nfib": n_fiber}
path = caching.get_cache_fpath(name, dependencies)

fibers = caching.read_cache(path)

name = "ipoints"
dependencies = {"nfib": n_fiber, "h": h}
path = caching.get_cache_fpath(name, dependencies)

if caching.is_cached(path):
    ipoints = caching.read_cache(path)
else:
    ipoints = calc_integration_points(egroup, shape)
    caching.write_cache(path, ipoints)

name = "distances"
dependencies = {"nfib": n_fiber, "h": h}
path = caching.get_cache_fpath(name, dependencies)

if caching.is_cached(path):
    distances = caching.read_cache(path)
else:
    distances = np.zeros(ipoints.shape[0])
    for ip, ipoint in enumerate(ipoints):
        fiber, dist = calc_closest_fiber(ipoint, fibers, 1.0)
        distances[ip] = dist
    caching.write_cache(path, distances)


backdoor = {}
backdoor["xcoord"] = ipoints[:, 0]
backdoor["ycoord"] = ipoints[:, 1]
backdoor["e"] = np.zeros(ipoints.shape[0])

E = 1000
decay = 20
reduction = 0.5

for ip, ipoint in enumerate(ipoints):
    dist = distances[ip]
    assert dist >= 0.09
    surf_dist = max(dist - 0.1, 0.0)
    moisture = np.exp(-decay * surf_dist)
    damage = reduction * moisture
    backdoor["e"][ip] = E * (1 - damage)


elem_stiffness = np.zeros(len(elems))

for group_name, egroup in groups.items():
    if group_name == "matrix":
        for ielem in egroup:
            inodes = elems[ielem]
            coords = nodes[inodes]
            midpoint = np.mean(coords, axis=0)
            fiber, dist = calc_closest_fiber(midpoint, fibers, 1.0)
            assert dist >= 0.09
            surf_dist = max(dist - 0.1, 0.0)
            moisture = np.exp(-decay * surf_dist)
            damage = reduction * moisture
            elem_stiffness[ielem] = E * (1 - damage)
    elif group_name == "fiber":
        ielems = egroup.get_indices()
        # elem_stiffness[ielems] = props["model"]["model"]["fiber"]["material"]["E"]
        elem_stiffness[ielems] = 0
    else:
        assert False


jive = CJiveRunner(props, elems=elems)
globdat = jive(**backdoor)


def calc_strains(globdat):
    state0 = globdat["state0"]
    elems = globdat["elemSet"]
    dofs = globdat["dofSpace"]
    shape = globdat["shape"]

    nodes = elems.get_nodes()

    ipcount = shape.ipoint_count()
    nodecount = shape.node_count()
    strains_xx = np.zeros((len(elems), ipcount))
    strains_yy = np.zeros((len(elems), ipcount))
    strains_xy = np.zeros((len(elems), ipcount))

    for ielem, inodes in enumerate(elems):
        idofs = dofs.get_dofs(inodes, ["dx", "dy"])
        eldisp = state0[idofs]

        coords = nodes[inodes]
        grads, wts = shape.get_shape_gradients(coords)

        for ip in range(ipcount):
            B = np.zeros((3, 2 * nodecount))

            for n in range(nodecount):
                B[0, 2 * n + 0] = grads[ip, 0, n]
                B[1, 2 * n + 1] = grads[ip, 1, n]
                B[2, 2 * n + 0] = grads[ip, 1, n]
                B[2, 2 * n + 1] = grads[ip, 0, n]

            ipstrain = B @ eldisp

            strains_xx[ielem, ip] = ipstrain[0]
            strains_yy[ielem, ip] = ipstrain[1]
            strains_xy[ielem, ip] = ipstrain[2]

    return strains_xx, strains_yy, strains_xy


eps_xx, eps_yy, gamma_xy = calc_strains(globdat)

eps_avg = 0.5 * (eps_xx + eps_yy)
eps_diff = 0.5 * (eps_xx - eps_yy)

eps_1 = eps_avg + np.sqrt(eps_diff**2 + (0.5 * gamma_xy) ** 2)
eps_2 = eps_avg - np.sqrt(eps_diff**2 + (0.5 * gamma_xy) ** 2)


from myjivex.util import ElemViewer

ElemViewer(
    abs(eps_1[:, 0]),
    globdat,
    maxcolor=0.01,
    title=r"max strain, $N_e = {}$".format(len(elems)),
)
ElemViewer(
    elem_stiffness,
    globdat,
    maxcolor=1000,
    title=r"stiffness, $N_e = {}$".format(len(elems)),
)
