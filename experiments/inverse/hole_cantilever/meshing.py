import os
import numpy as np
import gmsh


def create_mesh(*, h, L, H, x, y, a, theta, r_rel, h_meas, fname, n_refine=0, tol=1e-8):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.Verbosity", 2)  # only print errors and warnings

    gmsh.model.add("example.mesh")

    occ = gmsh.model.occ

    points = []
    points_with_dim = []
    for x_point in np.linspace(0, L, int(L / h_meas) + 1):
        for y_point in [0.0, H]:
            p = occ.addPoint(x_point, y_point, 0.0)
            points.append(p)
            points_with_dim.append((0, p))

    for y_point in np.linspace(0, H, int(H / h_meas) + 1)[1:-1]:
        x_point = L
        p = occ.addPoint(x_point, y_point, 0.0)
        points.append(p)
        points_with_dim.append((0, p))

    main_rect = occ.addRectangle(0.0, 0.0, 0.0, L, H)
    r = a * r_rel
    hole_rect = occ.addRectangle(x - 0.5 * a, y - 0.5 * a, 0.0, a, a, roundedRadius=r)

    main_with_dim = [(2, main_rect)]
    hole_with_dim = [(2, hole_rect)]

    occ.rotate(hole_with_dim, x, y, 0.0, 0.0, 0.0, 1.0, theta)
    diff = occ.cut(main_with_dim, hole_with_dim)[0]
    occ.fragment(points_with_dim, main_with_dim)[0]

    # Generate mesh
    occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [diff[0][1]])

    gmsh.option.setNumber("Mesh.MeshSizeMin", h)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)

    gmsh.model.mesh.generate(2)

    # Export in msh22 format
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(fname)

    # Refine the mesh n times
    for i in range(n_refine):
        gmsh.model.mesh.refine()
        name, ext = os.path.splitext(fname)
        rname = name + "-r" + str(i+1) + ext
        gmsh.write(rname)

    gmsh.model.remove()

    gmsh.finalize()
