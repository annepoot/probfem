import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from experiments.inverse.frp_damage import caching, misc, params


n_fiber = params.geometry_params["n_fiber"]
r_fiber = params.geometry_params["r_fiber"]
tol = params.geometry_params["tol_fiber"]
rve_size = params.geometry_params["rve_size"]
seed = params.geometry_params["seed_fiber"]

name = "fibers"
dependencies = {"nfib": n_fiber}
path = caching.get_cache_fpath(name, dependencies)

print("Computing fibers")
fibers = misc.calc_fibers(n=n_fiber, a=rve_size, r=r_fiber, tol=tol, seed=seed)
print("Writing fibers to cache")
caching.write_cache(path, fibers)

n_speckle = params.geometry_params["n_speckle"]
r_speckle = params.geometry_params["r_speckle"]
tol = params.geometry_params["tol_speckle"]
obs_size = params.geometry_params["obs_size"]
seed = params.geometry_params["seed_speckle"]

name = "speckles"
dependencies = {"nobs": n_speckle}
path = caching.get_cache_fpath(name, dependencies)

print("Computing speckles")
speckles = misc.calc_fibers(n=n_speckle, a=obs_size, r=r_speckle, tol=tol, seed=seed)
print("Writing speckles to cache")
caching.write_cache(path, speckles)

name = "connectivity"
dependencies = {"nobs": n_speckle}
path = caching.get_cache_fpath(name, dependencies)

print("Computing connectivity")
connectivity = misc.calc_connectivity(speckles=speckles)
print("Writing connectivity to cache")
caching.write_cache(path, connectivity)

fig, ax = plt.subplots()

for fiber in fibers:
    ax.add_patch(Circle(fiber, r_fiber, color="C0", alpha=0.5))

for speckle in speckles:
    ax.add_patch(Circle(speckle, 0.5 * r_speckle, color="C1", alpha=0.5))

for edge in connectivity:
    coords1, coords2 = speckles[edge]
    ax.plot([coords1[0], coords2[0]], [coords1[1], coords2[1]], color="C1", alpha=0.5)

ax.set_aspect("equal")

ax.set_xlim((-obs_size, obs_size))
ax.set_ylim((-obs_size, obs_size))
plt.show()

for h in [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
    print("Computing mesh for h =", h)
    fname = "meshes/rve_h-{:.3f}_nfib-{}.msh".format(h, n_fiber)
    misc.create_mesh(fibers=fibers, a=rve_size, r=r_fiber, h=h, fname=fname)
