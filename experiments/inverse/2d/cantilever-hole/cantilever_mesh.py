import numpy as np
import gmsh

occ = gmsh.model.occ


def create_mesh(*, lc, L, H, x, y, a, theta, r, tol=1e-8, fname="cantilever.msh"):
    gmsh.initialize()
    gmsh.model.add("example.mesh")

    points = []
    points_with_dim = []
    for x_point in np.linspace(0, L, 5):
        for y_point in np.linspace(0, H, 2):
            p = occ.addPoint(x_point, y_point, 0.0)
            points.append(p)
            points_with_dim.append((0, p))

    main_rect = occ.addRectangle(0.0, 0.0, 0.0, L, H)
    hole_rect = occ.addRectangle(x - 0.5 * a, y - 0.5 * a, 0.0, a, a, roundedRadius=r)

    main_with_dim = [(2, main_rect)]
    hole_with_dim = [(2, hole_rect)]

    occ.rotate(hole_with_dim, x, y, 0.0, 0.0, 0.0, 1.0, theta)
    diff = occ.cut(main_with_dim, hole_with_dim)[0]
    occ.fragment(points_with_dim, main_with_dim)[0]

    # Generate mesh
    occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [diff[0][1]])

    gmsh.option.setNumber("Mesh.MeshSizeMin", lc)
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc)

    gmsh.model.mesh.generate(2)

    # Export and finish
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(fname)
    gmsh.finalize()
