import numpy as np
import gmsh


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


n_fiber = 30
a = 1.0
r = 0.15
tol = 0.01
h = 0.01
seed = 0

fibers = calc_fibers(n_fiber=n_fiber, a=a, r=r, tol=tol, seed=seed)

np.save("meshes/rve_nfib-{}.npy".format(n_fiber), fibers)
fname = "meshes/rve_nfib-{}_h-{}.msh".format(n_fiber, h)

create_mesh(fibers=fibers, a=a, r=r, h=h, fname=fname)
