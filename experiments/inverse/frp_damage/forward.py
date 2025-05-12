import numpy as np

from fem.jive import CJiveRunner
from fem.meshing import read_mesh
from experiments.inverse.frp_damage.props import get_fem_props
from experiments.inverse.frp_damage import caching, misc, params

props = get_fem_props()

n_fiber = params.geometry_params["n_fiber"]
r_fiber = params.geometry_params["r_fiber"]
h = 0.05

fname = "meshes/rve_h-{:.3f}_nfib-{}.msh".format(h, n_fiber)
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

if caching.is_cached(path):
    ipoints = caching.read_cache(path)
else:
    ipoints = misc.calc_integration_points(egroup, shape)
    caching.write_cache(path, ipoints)

name = "distances"
dependencies = {"nfib": n_fiber, "h": h}
path = caching.get_cache_fpath(name, dependencies)

if caching.is_cached(path):
    distances = caching.read_cache(path)
else:
    distances = np.zeros(ipoints.shape[0])
    for ip, ipoint in enumerate(ipoints):
        fiber, dist = misc.calc_closest_fiber(ipoint, fibers, 1.0)
        distances[ip] = dist - r_fiber
    assert 0.0 < np.min(distances) < 0.1 * h
    caching.write_cache(path, distances)


backdoor = {}
backdoor["xcoord"] = ipoints[:, 0]
backdoor["ycoord"] = ipoints[:, 1]
backdoor["e"] = np.zeros(ipoints.shape[0])

E = params.material_params["E_matrix"]
alpha = params.material_params["alpha"]
beta = params.material_params["beta"]
c = params.material_params["c"]
d = params.material_params["d"]

for ip, ipoint in enumerate(ipoints):
    dist = distances[ip]
    sat = misc.saturation(dist, alpha, beta, c)
    dam = misc.damage(sat, d)
    backdoor["e"][ip] = E * (1 - dam)

elem_stiffness = np.zeros(len(elems))

for group_name, egroup in groups.items():
    if group_name == "matrix":
        for ie, ielem in enumerate(egroup):
            ip_stiffness = backdoor["e"][3 * ie : 3 * (ie + 1)]
            elem_stiffness[ielem] = np.mean(ip_stiffness)
    elif group_name == "fiber":
        ielems = egroup.get_indices()
        elem_stiffness[ielems] = 0
    else:
        assert False


jive = CJiveRunner(props, elems=elems, egroups=groups)
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

from myjivex.util import QuickViewer, ElemViewer

QuickViewer(
    globdat["state0"],
    globdat,
    comp=0,
)
ElemViewer(
    eps_xx[:, 0],
    globdat,
    maxcolor=0.01,
    title=r"max strain, $N_e = {}$".format(len(elems)),
)
ElemViewer(
    elem_stiffness,
    globdat,
    maxcolor=E,
    title=r"stiffness, $N_e = {}$".format(len(elems)),
)
