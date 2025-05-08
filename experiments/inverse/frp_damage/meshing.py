import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import Delaunay
import gmsh

from experiments.inverse.frp_damage import caching


def calc_fibers(*, n_fiber, a, r, tol=0.001, seed=0):
    rng = np.random.default_rng(seed)

    fiber_coords = []

    for i in range(n_fiber):
        for _ in range(int(1e5)):
            new_coords = rng.uniform(-a, a, size=2)

            fib, dist = calc_closest_fiber(new_coords, fiber_coords, a)

            if dist > 2 * r + tol:
                for x_off in [-2 * a, 0, 2 * a]:
                    x_mod = new_coords[0] + x_off

                    if x_mod < -a - r - tol or x_mod > a + r + tol:
                        continue

                    for y_off in [-2 * a, 0, 2 * a]:
                        y_mod = new_coords[1] + y_off

                        if y_mod < -a - r - tol or y_mod > a + r + tol:
                            continue

                        fiber_coords.append(np.array([x_mod, y_mod]))
                break
        else:
            print("fiber number:", i)
            raise RuntimeError("Number of tries exceeded (no fiber location found!)")

    return np.array(fiber_coords)


def create_mesh(*, fibers, a, r, h, fname):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.Verbosity", 0)  # only print errors and warnings

    gmsh.model.add("example.mesh")

    occ = gmsh.model.occ

    matrix_tag = occ.addRectangle(-a, -a, 0.0, 2 * a, 2 * a)
    matrix_dimtag = [(2, matrix_tag)]

    fiber_dimtags = []

    for fiber in fibers:
        fiber_tag = occ.addDisk(fiber[0], fiber[1], 0.0, r, r)
        fiber_dimtag = [(2, fiber_tag)]
        fiber_clipped = occ.intersect(fiber_dimtag, matrix_dimtag, removeTool=False)[0]
        fiber_dimtags.extend(fiber_clipped)

    fragments, _ = occ.fragment([(2, matrix_tag)], fiber_dimtags)

    # Generate mesh
    occ.synchronize()

    fiber_tags = []
    matrix_tags = []

    for dim, tag in fragments:
        box = gmsh.model.occ.getBoundingBox(dim, tag)
        size = box[3] - box[0]
        if size <= 2.2 * r:
            fiber_tags.append(tag)
        else:
            matrix_tags.append(tag)

    assert len(matrix_tags) == 1

    gmsh.model.addPhysicalGroup(2, matrix_tags)
    gmsh.model.setPhysicalName(2, 1, "matrix")
    gmsh.model.addPhysicalGroup(2, fiber_tags)
    gmsh.model.setPhysicalName(2, 2, "fiber")

    gmsh.option.setNumber("Mesh.MeshSizeMin", h)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)

    gmsh.model.mesh.generate(2)

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(fname)

    gmsh.model.remove()
    gmsh.finalize()


def calc_closest_fiber(point, fibers, rve_size):
    min_dist = np.inf
    argmin = -1

    ghost_points = np.zeros((9, 2))

    for ix, x_off in enumerate([-2 * rve_size, 0, 2 * rve_size]):
        for iy, y_off in enumerate([-2 * rve_size, 0, 2 * rve_size]):
            ghost_points[3 * ix + iy, 0] = point[0] + x_off
            ghost_points[3 * ix + iy, 1] = point[1] + y_off

    for i, fiber in enumerate(fibers):
        dist = np.min(np.sqrt(np.sum((ghost_points - fiber) ** 2, axis=1)))
        if dist < min_dist:
            min_dist = dist
            argmin = i

    return argmin, min_dist


def calc_connectivity(speckles, max_edge=0.1, min_ratio=2.1):
    tri = Delaunay(speckles)
    triangles = tri.simplices
    neighbors = tri.neighbors

    mask = np.ones(len(triangles), dtype=bool)

    for _ in range(100):
        pruned_something = False
        is_boundary = np.any(neighbors == -1, axis=1)

        for i, triangle in enumerate(triangles):
            if not mask[i]:
                continue

            if is_boundary[i]:
                vertices = speckles[triangle]
                edges = np.linalg.norm(vertices - np.roll(vertices, -1, axis=0), axis=1)

                if np.max(edges) > max_edge:
                    prune = True
                elif np.sum(edges) / np.max(edges) < min_ratio:
                    prune = True
                else:
                    prune = False

                if prune:
                    pruned_something = True
                    mask[i] = False

                    for neigh in neighbors[i]:
                        if neigh > -1:
                            neighbors[neigh][np.where(neighbors[neigh] == i)] = -1

        if not pruned_something:
            break

    return triangles[mask]


n_fiber = 30
rve_size = 1.0
r_fiber = 0.15
tol = 0.01
seed = 0

name = "fibers"
dependencies = {"nfib": n_fiber}
path = caching.get_cache_fpath(name, dependencies)

if caching.is_cached(path):
    fibers = caching.read_cache(path)
else:
    fibers = calc_fibers(n_fiber=n_fiber, a=rve_size, r=r_fiber, tol=tol, seed=seed)
    caching.write_cache(path, fibers)

n_obs = 200
obs_size = 0.3
r_speckle = 0.015
tol = 0.001
seed = 0

name = "speckles"
dependencies = {"nobs": n_obs}
path = caching.get_cache_fpath(name, dependencies)

if caching.is_cached(path):
    speckles = caching.read_cache(path)
else:
    speckles = calc_fibers(n_fiber=n_obs, a=obs_size, r=r_speckle, tol=tol, seed=seed)
    caching.write_cache(path, speckles)

name = "connectivity"
dependencies = {"nobs": n_obs}
path = caching.get_cache_fpath(name, dependencies)

if caching.is_cached(path):
    connectivity = caching.read_cache(path)
else:
    connectivity = calc_connectivity(speckles=speckles)
    caching.write_cache(path, connectivity)

fig, ax = plt.subplots()

for fiber in fibers:
    ax.add_patch(Circle(fiber, r_fiber, color="C0", alpha=0.5))

for speckle in speckles:
    ax.add_patch(Circle(speckle, 0.5 * r_speckle, color="C1", alpha=0.5))

ax.triplot(speckles[:, 0], speckles[:, 1], connectivity, color="C1", alpha=0.5)

ax.set_aspect("equal")
ax.set_xlim((-rve_size, rve_size))
ax.set_ylim((-rve_size, rve_size))
plt.show()

h = 0.05
fname = "meshes/rve_h-{}_nfib-{}.msh".format(h, n_fiber)

create_mesh(fibers=fibers, a=rve_size, r=r_fiber, h=h, fname=fname)
