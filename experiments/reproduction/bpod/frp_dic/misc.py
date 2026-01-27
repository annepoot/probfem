import os
import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import csr_array
import gmsh

from fem.meshing import (
    create_bbox,
    create_bboxes,
    find_coords_in_elementset,
    list_bbox_bbox_intersections,
    clip_polygons,
)


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


def create_mesh(*, fibers, a, r, h, fname, shift=False):

    if isinstance(h, str) and "r" in h:
        h, n_refine = h.split("r")
        h, n_refine = float(h), int(n_refine)
    else:
        n_refine = 0

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.Verbosity", 0)  # only print errors and warnings

    gmsh.model.add("example.mesh")

    occ = gmsh.model.occ

    matrix_tag = occ.addRectangle(-a, -a, 0.0, 2 * a, 2 * a)
    matrix_dimtag = [(2, matrix_tag)]

    fiber_dimtags = []

    assert isinstance(shift, bool)

    max_n_elem_per_arc = int(0.5 * np.pi * r / h + 0.5)

    n_poly = 4 * max_n_elem_per_arc
    area_polygon = 0.5 * n_poly * r**2 * np.sin(2 * np.pi / n_poly)
    area_circle = np.pi * r**2
    r_scaled = r * np.sqrt(area_circle / area_polygon)

    if shift:
        offset = 0.5 * np.pi / (2 * max_n_elem_per_arc)
    else:
        offset = 0.0

    # Create 4 points on the boundary, rotated if needed
    def boundary_point(i, *, x, y, r, offset):
        angle = i * np.pi / 2 + offset
        return occ.addPoint(x + r * np.cos(angle), y + r * np.sin(angle), 0)

    for fiber in fibers:
        x, y = fiber
        center = occ.addPoint(x, y, 0)
        p0 = boundary_point(0, x=x, y=y, r=r_scaled, offset=offset)  # (1,0)
        p1 = boundary_point(1, x=x, y=y, r=r_scaled, offset=offset)  # (0,1)
        p2 = boundary_point(2, x=x, y=y, r=r_scaled, offset=offset)  # (-1,0)
        p3 = boundary_point(3, x=x, y=y, r=r_scaled, offset=offset)  # (0,-1)

        # Create 4 circular arcs
        arc1 = occ.addCircleArc(p0, center, p1)
        arc2 = occ.addCircleArc(p1, center, p2)
        arc3 = occ.addCircleArc(p2, center, p3)
        arc4 = occ.addCircleArc(p3, center, p0)

        # Create loop and surface
        loop = occ.addCurveLoop([arc1, arc2, arc3, arc4])
        fiber_tag = occ.addPlaneSurface([loop])

        occ.synchronize()

        fiber_dimtag = [(2, fiber_tag)]
        fiber_clipped = occ.intersect(fiber_dimtag, matrix_dimtag, removeTool=False)[0]
        fiber_dimtags.extend(fiber_clipped)

    fragments, _ = occ.fragment([(2, matrix_tag)], fiber_dimtags)

    # Generate mesh
    occ.synchronize()

    fiber_tags = []
    matrix_tags = []

    for dim, tag in fragments:
        box = occ.getBoundingBox(dim, tag)
        size = box[3] - box[0]
        if size <= 2.2 * r_scaled:
            fiber_tags.append(tag)
        else:

            matrix_tags.append(tag)

    all_dimtags = [(2, tag) for tag in fiber_tags] + [(2, tag) for tag in matrix_tags]
    boundary = gmsh.model.getBoundary(
        all_dimtags, combined=False, oriented=False, recursive=False
    )

    for bdim, btag in boundary:
        btype = gmsh.model.getType(bdim, btag)

        if btype == "Line":
            _, endpoints = gmsh.model.getAdjacencies(bdim, btag)
            assert len(endpoints) == 2

            A = gmsh.model.getValue(0, endpoints[0], [])
            B = gmsh.model.getValue(0, endpoints[1], [])

            length = np.sqrt(np.sum((A - B) ** 2))

        elif btype == "Circle":
            _, endpoints = gmsh.model.getAdjacencies(bdim, btag)
            assert len(endpoints) == 2

            A = gmsh.model.getValue(0, endpoints[0], [])
            B = gmsh.model.getValue(0, endpoints[1], [])

            chord = np.sqrt(np.sum((A - B) ** 2))
            length = 2 * r * np.arcsin(0.5 * chord / r_scaled)
            assert 0.0 <= length <= 0.5 * np.pi * r + 1e-8

        else:
            raise ValueError("Unknown boundary type")

        n_elem = max(int(length / h + 0.5), 1)

        gmsh.model.mesh.setTransfiniteCurve(btag, n_elem + 1)

    occ.synchronize()

    assert len(matrix_tags) == 1

    gmsh.model.addPhysicalGroup(2, matrix_tags)
    gmsh.model.setPhysicalName(2, 1, "matrix")
    gmsh.model.addPhysicalGroup(2, fiber_tags)
    gmsh.model.setPhysicalName(2, 2, "fiber")

    gmsh.model.mesh.generate(2)

    if n_refine > 0:
        gmsh.write("tmp.msh")
    else:
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        gmsh.write(fname)

    gmsh.model.remove()
    gmsh.finalize()

    # necessary trick to hierarchically refine the mesh, even on curved boundaries
    if n_refine > 0:
        gmsh.initialize()
        gmsh.open("tmp.msh")

        for _ in range(n_refine):
            gmsh.model.mesh.refine()

        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        gmsh.write(fname)

        gmsh.finalize()

        os.remove("tmp.msh")


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
        idofs1 = dofs[inodes1]
        sfuncs1 = shape.eval_global_shape_functions(speckle1, coords1)

        speckle2 = speckles[ispeckle2]
        ielem2 = speckle_ielems[ispeckle2]
        inodes2 = elems[ielem2]
        coords2 = nodes[inodes2]
        idofs2 = dofs[inodes2]
        sfuncs2 = shape.eval_global_shape_functions(speckle2, coords2)

        rowidx.extend(np.full(6, 2 * i + 0))
        colidx.extend(idofs1[:, 0])
        values.extend(-sfuncs1)
        colidx.extend(idofs2[:, 0])
        values.extend(sfuncs2)

        rowidx.extend(np.full(6, 2 * i + 1))
        colidx.extend(idofs1[:, 1])
        values.extend(-sfuncs1)
        colidx.extend(idofs2[:, 1])
        values.extend(sfuncs2)

    n_obs = dofs.shape[1] * len(connectivity)
    n_dof = dofs.shape[0] * dofs.shape[1]

    return csr_array((values, (rowidx, colidx)), shape=(n_obs, n_dof))


def calc_dic_grid(h_dic, fibers, r_fiber, obs_size, rve_size):
    n = int(2 * obs_size / h_dic)

    grid = np.zeros((n**2, 4))

    for i in range(n):
        for j in range(n):
            x = i * h_dic - obs_size
            y = j * h_dic - obs_size
            w = h_dic
            h = h_dic
            grid[i * n + j] = [x, y, w, h]

    mask = np.zeros(n**2, dtype=bool)
    point = np.zeros(2)

    for i, square in enumerate(grid):
        dists = []
        for dx in [0, 1]:
            for dy in [0, 1]:
                point[0] = square[0] + dx * square[2]
                point[1] = square[1] + dy * square[3]
                dists.append(calc_closest_fiber(point, fibers, rve_size)[1])

        dist = np.max(dists)

        if dist > r_fiber:
            mask[i] = True

    return grid[mask]


def calc_dic_operator(fibers, grid, elems, dofs, shape):
    nodes = elems.get_nodes()
    elem_bboxes = create_bboxes(elems)

    def polygon_area(coords):
        area = 0.0
        n = len(coords)

        for i in range(n):
            a = coords[i]
            b = coords[(i + 1) % n]
            area += 0.5 * (a[0] * b[1] - a[1] * b[0])

        return area

    n_obs = 3 * len(grid)
    n_dof = dofs.shape[0] * dofs.shape[1]

    rowidx = []
    colidx = []
    values = []

    for i, square in enumerate(grid):
        square_coords = np.zeros((4, 2))
        square_coords[0] = [square[0] + 0.0 * square[2], square[1] + 0.0 * square[3]]
        square_coords[1] = [square[0] + 1.0 * square[2], square[1] + 0.0 * square[3]]
        square_coords[2] = [square[0] + 1.0 * square[2], square[1] + 1.0 * square[3]]
        square_coords[3] = [square[0] + 0.0 * square[2], square[1] + 1.0 * square[3]]

        square_bbox = create_bbox(square_coords)

        ielems = list_bbox_bbox_intersections(square_bbox, elem_bboxes)

        clips = []
        rel_areas = []

        for ielem in ielems:
            inodes = elems[ielem]
            elem_coords = nodes[inodes]
            idofs = dofs[inodes]

            clip = clip_polygons(elem_coords, square_coords)

            if len(clip) == 0:
                continue

            area = polygon_area(clip)
            rel_area = area / (square[2] * square[3])

            clips.append(clip)
            rel_areas.append(rel_area)

            grads, wts = shape.get_shape_gradients(elem_coords)

            # constant strain, so only use one integration point
            assert np.allclose(grads[0], grads[1])
            assert np.allclose(grads[0], grads[2])

            rowidx.extend(np.full(3, 3 * i + 0))
            colidx.extend(idofs[:, 0])
            values.extend(rel_area * grads[0, 0])

            rowidx.extend(np.full(3, 3 * i + 1))
            colidx.extend(idofs[:, 1])
            values.extend(rel_area * grads[0, 1])

            rowidx.extend(np.full(3, 3 * i + 2))
            colidx.extend(idofs[:, 0])
            values.extend(rel_area * grads[0, 1])

            rowidx.extend(np.full(3, 3 * i + 2))
            colidx.extend(idofs[:, 1])
            values.extend(rel_area * grads[0, 0])

    return csr_array((values, (rowidx, colidx)), shape=(n_obs, n_dof))


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


def calc_elem_stiffnesses(ip_stiffnesses, egroups):
    elems = next(iter(egroups.values())).get_elements()
    elem_stiffnesses = np.zeros(len(elems))

    for group_name, egroup in egroups.items():
        if group_name == "matrix":
            for ie, ielem in enumerate(egroup):
                ip_stiffness = ip_stiffnesses[3 * ie : 3 * (ie + 1)]
                elem_stiffness = np.mean(ip_stiffness)
                elem_stiffnesses[ielem] = elem_stiffness
        elif group_name == "fiber":
            ielems = egroup.get_indices()
            elem_stiffnesses[ielems] = 0
        else:
            assert False

    return elem_stiffnesses


def sigmoid(x, decay, shift):
    return 1.0 / (1 + np.exp(-decay * (x - shift)))


def saturation(x, alpha, beta, c):
    def base_func(x):
        return (1 - sigmoid(x, alpha, c)) * sigmoid(x, beta, 0.0)

    const = base_func(0.0)
    return base_func(x) / const


def damage(saturation, d):
    return d * saturation


def calc_damage_map(ipoints, distances, domain):
    # maps damage in 1D domain to damage per integration point
    rowidx = []
    colidx = []
    values = []

    input_map = (len(domain) - 1) / np.max(domain)

    for ip, ipoint in enumerate(ipoints):
        dist = max(distances[ip], 0.0)
        idx_l = int(dist * input_map)
        idx_r = idx_l + 1

        x_l = domain[idx_l]
        x_r = domain[idx_r]

        rowidx.append(ip)
        rowidx.append(ip)
        colidx.append(idx_l)
        colidx.append(idx_r)
        values.append(1.0 - (dist - x_l) / (x_r - x_l))
        values.append(1.0 - (x_r - dist) / (x_r - x_l))

    valrowcol = (values, (rowidx, colidx))
    shape = (len(ipoints), len(domain))

    return csr_array(valrowcol, shape=shape)
