import os
import numpy as np

from myjive.fem import Tri3Shape
from myjivex.util import QuickViewer, ElemViewer

from fem.jive import CJiveRunner
from fem.meshing import read_mesh

from experiments.inverse.frp_damage.props import get_fem_props
from experiments.inverse.frp_damage import caching, misc, params

rve_size = params.geometry_params["rve_size"]
n_fiber = params.geometry_params["n_fiber"]
r_fiber = params.geometry_params["r_fiber"]
tol = params.geometry_params["tol_fiber"]
seed = params.geometry_params["seed_fiber"]

#######################
# get fiber locations #
#######################

name = "fibers"
dependencies = {"nfib": n_fiber}
path = caching.get_cache_fpath(name, dependencies)

if caching.is_cached(path):
    print("Reading fibers from cache")
    fibers = caching.read_cache(path)
else:
    print("Computing fibers")
    fibers = misc.calc_fibers(n=n_fiber, a=rve_size, r=r_fiber, tol=tol, seed=seed)
    print("Writing fibers to cache")
    caching.write_cache(path, fibers)

############
# get mesh #
############

h = 0.002

fname = "meshes/rve_h-{}_nfib-{}.msh".format(h, n_fiber)

if not os.path.exists(fname):
    print("Computing mesh")
    misc.create_mesh(fibers=fibers, a=rve_size, r=r_fiber, h=h, fname=fname)

##########################
# get integration points #
##########################

nodes, elems, groups = read_mesh(fname, read_groups=True)
egroup = groups["matrix"]
shape = Tri3Shape("Gauss3")

name = "ipoints"
dependencies = {"nfib": n_fiber, "h": h}
path = caching.get_cache_fpath(name, dependencies)

if caching.is_cached(path):
    print("Reading ipoints from cache")
    ipoints = caching.read_cache(path)
else:
    print("Computing ipoints")
    ipoints = misc.calc_integration_points(egroup, shape)
    print("Writing ipoints to cache")
    caching.write_cache(path, ipoints)

#######################
# get fiber distances #
#######################

name = "distances"
dependencies = {"nfib": n_fiber, "h": h}
path = caching.get_cache_fpath(name, dependencies)

if caching.is_cached(path):
    print("Reading distances from cache")
    distances = caching.read_cache(path)
else:
    print("Computing distances")
    distances = np.zeros(ipoints.shape[0])
    for ip, ipoint in enumerate(ipoints):
        fiber, dist = misc.calc_closest_fiber(ipoint, fibers, 1.0)
        distances[ip] = dist - r_fiber
    assert 0.0 < np.min(distances) < 0.1 * h
    print("Writing distances to cache")
    caching.write_cache(path, distances)

###################
# get stiffnesses #
###################

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

#####################
# get displacements #
#####################

props = get_fem_props()
props["userinput"]["gmsh"]["file"] = fname

print("Computing solution")

jive = CJiveRunner(props, elems=elems)
globdat = jive(**backdoor)

QuickViewer(
    globdat["state0"],
    globdat,
    comp=0,
)
ElemViewer(
    elem_stiffness,
    globdat,
    maxcolor=E,
    title=r"stiffness, $N_e = {}$".format(len(elems)),
)


#########################
# get speckle locations #
#########################

obs_size = params.geometry_params["obs_size"]
n_speckle = params.geometry_params["n_speckle"]
r_speckle = params.geometry_params["r_speckle"]
tol = params.geometry_params["tol_speckle"]
seed = seed = params.geometry_params["seed_speckle"]

name = "speckles"
dependencies = {"nobs": n_speckle}
path = caching.get_cache_fpath(name, dependencies)

if caching.is_cached(path):
    print("Reading speckles from cache")
    speckles = caching.read_cache(path)
else:
    print("Computing speckles")
    speckles = misc.calc_fibers(
        n=n_speckle, a=obs_size, r=r_speckle, tol=tol, seed=seed
    )
    print("Writing speckles to cache")
    caching.write_cache(path, speckles)

#########################
# get speckle neighbors #
#########################

name = "connectivity"
dependencies = {"nobs": n_speckle}
path = caching.get_cache_fpath(name, dependencies)

if caching.is_cached(path):
    print("Reading connectivity from cache")
    connectivity = caching.read_cache(path)
else:
    print("Computing connectivity")
    connectivity = misc.calc_connectivity(speckles)
    print("Writing connectivity to cache")
    caching.write_cache(path, connectivity)

############################
# get observation operator #
############################

name = "observer"
dependencies = {"nobs": n_speckle, "h": h, "nfib": n_fiber, "sparse": True}
path = caching.get_cache_fpath(name, dependencies)

if caching.is_cached(path):
    print("Reading observer from cache")
    obs_operator = caching.read_cache(path)
else:
    print("Computing observer")
    dofs = globdat["dofSpace"]
    obs_operator = misc.calc_observer(speckles, connectivity, elems, dofs, shape)
    print("Writing observer to cache")
    caching.write_cache(path, obs_operator)

####################
# get ground truth #
####################

name = "truth"
dependencies = {"nobs": n_speckle, "h": h, "nfib": n_fiber}
path = caching.get_cache_fpath(name, dependencies)

if caching.is_cached(path):
    print("Reading truth from cache")
    truth = caching.read_cache(path)
else:
    print("Computing truth")
    truth = obs_operator @ globdat["state0"]
    print("Writing truth to cache")
    caching.write_cache(path, truth)


import matplotlib.pyplot as plt
import matplotlib as mpl

comp = 0
edgedisp = truth[comp::2]
norm = mpl.colors.Normalize(vmin=np.min(edgedisp), vmax=np.max(edgedisp))
cmap = mpl.colormaps["viridis"]

fig, ax = plt.subplots()

for i, edge in enumerate(connectivity):
    coord1, coord2 = speckles[edge]
    dl = edgedisp[i]

    color = cmap(norm(dl))
    ax.plot([coord1[0], coord2[0]], [coord1[1], coord2[1]], color=color)

ax.set_aspect("equal")
ax.set_xlim((-obs_size, obs_size))
ax.set_ylim((-obs_size, obs_size))
plt.show()
