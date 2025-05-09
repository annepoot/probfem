import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import csr_array
import gmsh

from fem.meshing import create_bboxes, find_coords_in_elementset


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


def calc_fibers(*, n, a, r, tol=0.001, seed=0):
    rng = np.random.default_rng(seed)

    fiber_coords = []

    for i in range(n):
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

    triangles = triangles[mask]

    edges = set()

    for triangle in triangles:
        for i, j in [(0, 1), (1, 2), (2, 0)]:
            vertex_i = triangle[i]
            vertex_j = triangle[j]

            assert vertex_i >= 0 and vertex_j >= 0 and vertex_i != vertex_j

            if vertex_i < vertex_j:
                edges.add((vertex_i, vertex_j))
            else:
                edges.add((vertex_j, vertex_i))

    return np.array(list(edges))


def calc_observer(speckles, connectivity, elems, dofs, shape):
    speckle_ielems = np.zeros(len(speckles), dtype=int)
    nodes = elems.get_nodes()
    bboxes = create_bboxes(elems)

    for i, speckle in enumerate(speckles):
        speckle_ielems[i] = find_coords_in_elementset(speckle, elems, bboxes=bboxes)

    rowidx = []
    colidx = []
    values = []

    for i, edge in enumerate(connectivity):
        ispeckle1, ispeckle2 = edge

        speckle1 = speckles[ispeckle1]
        ielem1 = speckle_ielems[ispeckle1]
        inodes1 = elems[ielem1]
        coords1 = nodes[inodes1]
        idofs1 = dofs.get_dofs(inodes1, ["dx", "dy"])
        sfuncs1 = shape.eval_global_shape_functions(speckle1, coords1)

        speckle2 = speckles[ispeckle2]
        ielem2 = speckle_ielems[ispeckle2]
        inodes2 = elems[ielem2]
        coords2 = nodes[inodes2]
        idofs2 = dofs.get_dofs(inodes2, ["dx", "dy"])
        sfuncs2 = shape.eval_global_shape_functions(speckle2, coords2)

        rowidx.extend(np.full(6, 2 * i + 0))
        colidx.extend(idofs1[0::2])
        values.extend(-sfuncs1)
        colidx.extend(idofs2[0::2])
        values.extend(sfuncs2)

        rowidx.extend(np.full(6, 2 * i + 1))
        colidx.extend(idofs1[1::2])
        values.extend(-sfuncs1)
        colidx.extend(idofs2[1::2])
        values.extend(sfuncs2)

    n_obs = 2 * len(connectivity)
    n_dof = dofs.dof_count()

    return csr_array((values, (rowidx, colidx)), shape=(n_obs, n_dof))


def sigmoid(x, decay, shift):
    return 1.0 / (1 + np.exp(-decay * (x - shift)))


def saturation(x, alpha, beta, c):
    def base_func(x):
        return (1 - sigmoid(x, alpha, c)) * sigmoid(x, beta, 0.0)

    const = base_func(0.0)
    return base_func(x) / const


def damage(saturation, d):
    return d * saturation
