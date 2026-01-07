import numpy as np
from fem.meshing import     list_bbox_bbox_intersections,    clip_polygons
from warnings import warn


###################################
# x- and y-components of solution #
###################################


def true_f_ux(x):
    return (1 - x**2) * (np.exp(2 * x) - 1)


def true_f_uy(y):
    return np.sin(np.pi * y)


def true_f_dux_dx(x):
    return 2 * x + np.exp(2 * x) * (2 - 2 * x - 2 * x**2)


def true_f_duy_dy(y):
    return np.pi * np.cos(np.pi * y)


def true_f_d2ux_dx2(x):
    return 2 + np.exp(2 * x) * (2 - 8 * x - 4 * x**2)


def true_f_d2uy_dy2(y):
    return -np.pi**2 * np.sin(np.pi * y)


###########################
# full 1d and 2d solution #
###########################


def true_f_solution_1d(coord):
    assert coord.shape[-1] == 1
    x = coord[..., 0]
    return true_f_ux(x)


def true_f_strain_1d(coord):
    assert coord.shape[-1] == 1
    x = coord[..., 0]
    return true_f_dux_dx(x)


def true_f_source_1d(coord):
    assert coord.shape[-1] == 1
    x = coord[..., 0]
    return -true_f_d2ux_dx2(x)


def true_f_solution_2d(coord):
    assert coord.shape[-1] == 2
    x = coord[..., 0]
    y = coord[..., 1]

    ux = true_f_ux(x)
    uy = true_f_uy(y)
    return ux * uy


def true_f_strain_2d(coord):
    assert coord.shape[-1] == 2
    x = coord[..., 0]
    y = coord[..., 1]

    ux = true_f_ux(x)
    uy = true_f_uy(y)
    dux_dx = true_f_dux_dx(x)
    duy_dy = true_f_duy_dy(y)
    return np.array([dux_dx * uy, ux * duy_dy])


def true_f_source_2d(coord):
    assert coord.shape[-1] == 2
    x = coord[..., 0]
    y = coord[..., 1]

    ux = true_f_ux(x)
    uy = true_f_uy(y)
    d2ux_dx2 = true_f_d2ux_dx2(x)
    d2uy_dy2 = true_f_d2uy_dy2(y)
    return -(d2ux_dx2 * uy + ux * d2uy_dy2)


###########################################
# x- and y-components of adjoint solution #
###########################################


def true_g_d2ux_dx2(x, ax, bx):
    if x < ax:
        return 0.0
    elif x > bx:
        return 0.0
    else:
        dx = bx - ax
        return -1 / dx


def true_g_d2uy_dy2(y, ay, by):
    if y < ay:
        return 0.0
    elif y > by:
        return 0.0
    else:
        dy = by - ay
        return -1 / dy


def true_g_dux_dx(x, ax, bx):
    mx = 0.5 * (ax + bx)
    dx = bx - ax
    A = 0.5 - mx
    B = mx * (1 - mx) - dx / 8

    if x < ax:
        return 1 - mx
    elif x > bx:
        return -mx
    else:
        sx = x - mx
        return -sx / dx + A


def true_g_duy_dy(y, ay, by):
    my = 0.5 * (ay + by)
    dy = by - ay
    A = 0.5 - my
    B = my * (1 - my) - dy / 8

    if y < ay:
        return 1 - my
    elif y > by:
        return -my
    else:
        sy = y - my
        return -sy / dy + A


def true_g_ux(x, ax, bx):
    mx = 0.5 * (ax + bx)
    dx = bx - ax
    A = 0.5 - mx
    B = mx * (1 - mx) - dx / 8

    if x < ax:
        return x * (1 - mx)
    elif x > bx:
        return (1 - x) * mx
    else:
        return -(x - ax) * (x - bx) / (2 * dx) + (0.5 - mx) * x + 0.5 * ax


def true_g_uy(y, ay, by):
    my = 0.5 * (ay + by)
    dy = by - ay
    A = 0.5 - my
    B = my * (1 - my) - dy / 8

    if y < ay:
        return y * (1 - my)
    elif y > by:
        return (1 - y) * my
    else:
        return -(y - ay) * (y - by) / (2 * dy) + (0.5 - my) * y + 0.5 * ay


###################################
# full 1d and 2d adjoint solution #
###################################


def true_g_source_1d(coord, a, b):
    assert coord.shape[-1] == 1
    x = coord[..., 0]
    ax, bx = a[0], b[0]
    return -true_g_d2ux_dx2(x, ax, bx)


def true_g_strain_1d(coord, a, b):
    assert coord.shape[-1] == 1
    x = coord[..., 0]
    ax, bx = a[0], b[0]
    return true_g_dux_dx(x, ax, bx)


def true_g_solution_1d(coord, a, b):
    assert coord.shape[-1] == 1
    x = coord[..., 0]
    ax, bx = a[0], b[0]
    return true_g_ux(x, ax, bx)


def true_g_source_2d(coord, a, b):
    assert coord.shape[-1] == 2
    x = coord[..., 0]
    y = coord[..., 1]
    ax, ay = a
    bx, by = b

    ux = true_g_ux(x, ax, bx)
    uy = true_g_uy(y, ay, by)
    d2ux_dx2 = true_g_d2ux_dx2(x, ax, bx)
    d2uy_dy2 = true_g_d2uy_dy2(y, ay, by)

    return -(d2ux_dx2 * uy + ux * d2uy_dy2)


def true_g_strain_2d(coord, a, b):
    assert coord.shape[-1] == 2
    x = coord[..., 0]
    y = coord[..., 1]
    ax, ay = a
    bx, by = b

    ux = true_g_ux(x, ax, bx)
    uy = true_g_uy(y, ay, by)
    dux_dx = true_g_dux_dx(x, ay, by)
    duy_dy = true_g_duy_dy(y, ay, by)
    return np.array([dux_dx * uy, ux * duy_dy])


def true_g_solution_2d(coord, a, b):
    assert coord.shape[-1] == 2
    x = coord[..., 0]
    y = coord[..., 1]
    ax, ay = a
    bx, by = b

    ux = true_g_ux(x, ax, bx)
    uy = true_g_uy(y, ay, by)
    return ux * uy


############################################
# l2- and energy norms of adjoint solution #
############################################


def true_g_ux_H0_norm(ax, bx):
    mx = 0.5 * (ax + bx)
    dx = bx - ax
    A = 0.5 - mx
    B = mx * (1 - mx) - dx / 8

    # int_0^a -> 1/3 (1 - m)**2 a**3
    # int_b^1 -> 1/3 m**2 (1 - b)**3
    # int_a^b -> 1/320 d**3 + 1/12 A**2 d**3 - 1/12 B d**2 + B**2 d
    eval_0a = (1 - mx) ** 2 * ax**3 / 3
    eval_ab = dx**3 / 320 + A**2 * dx**3 / 12 - B * dx**2 / 12 + B**2 * dx
    eval_b1 = mx**2 * (1 - bx) ** 3 / 3
    return np.sqrt(eval_0a + eval_ab + eval_b1)


def true_g_ux_H1_seminorm(ax, bx):
    mx = 0.5 * (ax + bx)
    dx = bx - ax
    return np.sqrt(mx * (1 - mx) - dx / 6)


def true_g_uy_H0_norm(ay, by):
    # same as ux norm due to symmetry
    ax, bx = ay, by
    return true_g_ux_H0_norm(ax, bx)


def true_g_uy_H1_seminorm(ay, by):
    # same as ux norm due to symmetry
    ax, bx = ay, by
    return true_g_ux_H1_seminorm(ax, bx)


#########################
# fem adjoint solutions #
#########################


def fem_g_source(a, b, *, globdat):
    nodes = globdat["nodeSet"]
    elems = globdat["elemSet"]
    bboxes = globdat["bboxes"]
    dofs = globdat["dofSpace"]
    shape = globdat["shape"]

    assert nodes is elems.get_nodes()

    dof_types = dofs.get_types()
    g = np.zeros(dofs.dof_count())

    if nodes.rank() == 1:
        bbox_ab = (np.array([a[0]]), np.array([b[0]]))
        ielems = list_bbox_bbox_intersections(bboxes, bbox_ab)

        dx = b[0] - a[0]
        height = 1 / dx

        for ielem in ielems:
            inodes = elems[ielem]
            coords = nodes[inodes]
            idofs = dofs.get_dofs(inodes, dof_types)

            # get intersection
            line = np.array([[max(a[0], coords[0, 0])], [min(b[0], coords[1, 0])]])
            ipoint = np.mean(line, axis=0)
            area = np.abs(line[1] - line[0])
            loc_point = shape.get_local_point(ipoint, coords)
            sfuncs = shape.eval_shape_functions(loc_point)
            g[idofs] += area * height * sfuncs

        return g

    elif nodes.rank() == 2:
        dx = b[0] - a[0]
        dy = b[1] - a[1]

        bbox_ab_ab = (np.array([a[0], a[1]]), np.array([b[0], b[1]]))
        bbox_0a_ab = (np.array([0.0, a[1]]), np.array([a[0], b[1]]))
        bbox_b1_ab = (np.array([b[0], a[1]]), np.array([1.0, b[1]]))
        bbox_ab_0a = (np.array([a[0], 0.0]), np.array([b[0], a[1]]))
        bbox_ab_b1 = (np.array([a[0], b[1]]), np.array([b[0], 1.0]))

        poly_ab_ab = np.array([[a[0], a[1]], [b[0], a[1]], [b[0], b[1]], [a[0], b[1]]])
        poly_0a_ab = np.array([[0.0, a[1]], [a[0], a[1]], [a[0], b[1]], [0.0, b[1]]])
        poly_b1_ab = np.array([[b[0], a[1]], [1.0, a[1]], [1.0, b[1]], [b[0], b[1]]])
        poly_ab_0a = np.array([[a[0], 0.0], [b[0], 0.0], [b[0], a[1]], [a[0], a[1]]])
        poly_ab_b1 = np.array([[a[0], b[1]], [b[0], b[1]], [b[0], 1.0], [a[0], 1.0]])

        bbox_list = [bbox_ab_ab, bbox_0a_ab, bbox_b1_ab, bbox_ab_0a, bbox_ab_b1]
        poly_list = [poly_ab_ab, poly_0a_ab, poly_b1_ab, poly_ab_0a, poly_ab_b1]
        var_list = ["xy", "x", "x", "y", "y"]

        for i, (bbox, poly, var) in enumerate(zip(bbox_list, poly_list, var_list)):
            ielems = list_bbox_bbox_intersections(bboxes, bbox)

            for ielem in ielems:
                elem_bbox = (bboxes[0][ielem], bboxes[1][ielem])
                inodes = elems[ielem]
                coords = nodes[inodes]
                idofs = dofs.get_dofs(inodes, dof_types)

                # get convex polygon intersection
                if np.all(elem_bbox[0] > bbox[0]) and np.all(elem_bbox[1] < bbox[1]):
                    clip = coords
                else:
                    clip = clip_polygons(coords, poly)

                # skip element if no intersection is found
                if len(clip) == 0:
                    continue

                # fan triangulation
                for ic in range(0, len(clip) - 2):
                    tri = clip[[ic, ic + 1, -1]]
                    ipoint = np.mean(tri, axis=0)
                    area = 0.5 * np.linalg.det(np.hstack([tri, np.ones((3, 1))]))

                    loc_point = shape.get_local_point(ipoint, coords)
                    sfuncs = shape.eval_shape_functions(loc_point)

                    if "x" in var:
                        d2uy_dy2 = true_g_d2uy_dy2(ipoint[1], a[1], b[1])
                        ux = true_g_ux(ipoint[0], a[0], b[0])
                        assert d2uy_dy2 < 0.0
                        assert ux > 0.0
                        g[idofs] -= area * d2uy_dy2 * ux * sfuncs

                    if "y" in var:
                        d2ux_dx2 = true_g_d2ux_dx2(ipoint[0], a[0], b[0])
                        uy = true_g_uy(ipoint[1], a[1], b[1])
                        assert d2ux_dx2 < 0.0
                        assert uy > 0.0
                        g[idofs] -= area * d2ux_dx2 * uy * sfuncs

        return g

    else:
        assert False


def fem_g_solution(a, b, *, globdat, g=None):
    nodes = globdat["nodeSet"]
    elems = globdat["elemSet"]
    dofs = globdat["dofSpace"]
    shape = globdat["shape"]

    assert nodes is elems.get_nodes()

    if g is None:
        g = fem_prior_source(a, b, globdat=globdat)

    Kc = globdat["Kc"]
    cdofs = globdat["constraints"].get_constraints()[0]
    g[cdofs] = 0.0

    ug = Kc.solve_A(g)
    return ug


def fem_quadrature(func, *, mesh, dofs, shape, args={}):
    nodes, elems = mesh
    dof_types = dofs.get_types()
    type_count = len(dof_types)
    dof_count = shape.node_count() * type_count
    integral = np.zeros(dofs.dof_count())

    N = np.zeros((type_count, dof_count))
    b = np.zeros(type_count)
    elint = np.zeros(dof_count)

    warn("assuming all elements have exactly the same shape")
    sfuncs = shape.get_shape_functions()
    coords_0 = nodes[elems[0]]
    iwts = shape.get_integration_weights(coords_0)
    ip_count = len(iwts)
    Ns = np.zeros((ip_count, type_count, dof_count))

    for ip in range(len(iwts)):
        for i in range(type_count):
            N[i, i::type_count] = sfuncs[ip]
        Ns[ip] = N

    for ielem, inodes in enumerate(elems):
        coords = nodes[inodes]
        idofs = dofs.get_dofs(inodes, types=dof_types)
        ipoints = shape.get_global_integration_points(coords)

        elint[:] = 0.0

        for ip, (ipoint, iwt) in enumerate(zip(ipoints, iwts)):
            for i in range(type_count):
                # N[i, i::type_count] = sfuncs[ip]
                b[i] = func(ipoint, **args)

            elint += iwt * Ns[ip].T @ b
        integral[idofs] += elint

    return integral


########################################################
# inner products between solution and adjoint solution #
########################################################


def true_integral_ux_ugx(ax, bx):
    mx = 0.5 * (ax + bx)
    dx = bx - ax

    def I1(x):
        # I1 = int x * (1 - mx) * (1 - x**2) * (exp(2 * x) - 1) dx
        #    = (mx - 1) / 8 * (-2 * x**4 + 4 * x**2 + exp(2 * x) * (4 * x**3 - 6 * x**2 + 2 * x - 1))
        return (
            0.125
            * (mx - 1)
            * (-2 * x**4 + 4 * x**2 + np.exp(2 * x) * (4 * x**3 - 6 * x**2 + 2 * x - 1))
        )

    def I2(x):
        # I2 = int ((x - ax) * (x - bx) / (2 * dx) + (0.5 - mx) * x + 0.5 * ax) * (1 - x**2) * (exp(2 * x) - 1) dx
        #    = int (-x**2 / (2 * dx) + (mx / dx + 0.5 - mx) * x + 0.5 * ax - ax * bx / (2 * dx)) * (1 - x**2) * (exp(2 * x) - 1) dx
        #    = int (alpha x**2 + beta x + gamma) * (1 - x**2) * (exp(2 * x) - 1) dx
        #    = alpha x**5 / 5 + beta x**4 / 4 + x**3 (gamma - alpha) / 3 - alpha / 2 * x**2
        alpha = -0.5 / dx
        beta = mx / dx + 0.5 - mx
        gamma = 0.5 * ax - (ax * bx) / (2 * dx)

        exp_part = (
            0.125
            * np.exp(2 * x)
            * (
                beta
                + 2 * gamma
                - 4 * beta * x**3
                - 4 * alpha * (x - 1) ** 2 * (x**2 + 1)
                + 6 * beta * x**2
                - 4 * gamma * x**2
                - 2 * beta * x
                + 4 * gamma * x
            )
        )
        pol_part = (
            x
            / 60
            * (
                -60 * gamma
                + 12 * alpha * x**4
                + 15 * beta * x**3
                - 20 * x**2 * (alpha - gamma)
                - 30 * beta * x
            )
        )

        return exp_part + pol_part

    def I3(x):
        # I3 = int (1 - x) * mx * (1 - x**2) * (exp(2 * x) - 1) dx
        #    = m / 24 * (-2 * (x - 1)**3 * (3 * x + 5) + 3 * exp(2 * x) * (4 * x**3 - 10 * x**2 + 6 * x + 1))
        return (
            mx
            / 24
            * (
                3 * np.exp(2 * x) * (4 * x**3 - 10 * x**2 + 6 * x + 1)
                - 2 * (x - 1) ** 3 * (3 * x + 5)
            )
        )

    # from scipy.integrate import quad

    # def integrand(x):
    #     return true_f_ux(x) * true_g_ux(x, ax, bx)

    # print(quad(integrand, 0.0, ax)[0], I1(ax) - I1(0.0))
    # print(quad(integrand, ax, bx)[0], I2(bx) - I2(ax))
    # print(quad(integrand, bx, 1.0)[0], I3(1.0) - I3(bx))

    # num_approx = quad(integrand, 0, 1)[0]
    # print(num_approx, I1(ax) - I1(0.0) + I2(bx) - I2(ax) + I3(1.0) - I3(bx))

    return I1(ax) - I1(0.0) + I2(bx) - I2(ax) + I3(1.0) - I3(bx)


def true_integral_uy_ugy(ay, by):
    my = 0.5 * (ay + by)
    dy = by - ay

    def I1(y):
        # I1 = int y * (1 - my) * sin(pi * y) dy
        #    = (1 - my) * (sin(pi * y) / pi**2 - y * cos(pi * y) / pi)
        return (1 - my) * (np.sin(np.pi * y) / np.pi**2 - y * np.cos(np.pi * y) / np.pi)

    def I2(y):
        # I2 = int ((y - ay) * (y - by) / (2 * dy) + (0.5 - my) * y + 0.5 * ay) * sin(pi * y) dy
        #    = int (-y**2 / (2 * dy) + (my / dy + 0.5 - my) * y + 0.5 * ay - ay * by / (2 * dy)) * sin(pi * y) dy
        #    = int (alpha * y**2 + beta * y + gamma) * sin(pi * y) dy
        #    = (y^2 / (2 * dy * pi) - alpha * y / pi - beta / pi - 1 / (dy * pi**3)) * cos(pi * x)
        #      + (-x / (dy * pi**2) + alpha / pi**2) * sin(pi * x)
        alpha = -0.5 / dy
        beta = my / dy + 0.5 - my
        gamma = 0.5 * ay - (ay * by) / (2 * dy)
        cos_part = (
            -np.cos(np.pi * y)
            / np.pi**3
            * (
                -2 * alpha
                + np.pi**2 * alpha * y**2
                + np.pi**2 * beta * y
                + np.pi**2 * gamma
            )
        )
        sin_part = np.sin(np.pi * y) / np.pi**2 * (beta + 2 * alpha * y)

        return cos_part + sin_part

    def I3(y):
        # I3 = int (1 - y) * my * sin(pi * y) dy
        #    = -my * (sin(pi * y) / pi**2 + (1 - y) * cos(pi * y) / pi)
        return -my * (
            np.sin(np.pi * y) / np.pi**2 + (1 - y) * np.cos(np.pi * y) / np.pi
        )

    # from scipy.integrate import quad

    # def integrand(y):
    #     return true_f_uy(y) * true_g_uy(y, ay, by)

    # print(quad(integrand, 0.0, ay)[0], I1(ay) - I1(0.0))
    # print(quad(integrand, ay, by)[0], I2(by) - I2(ay))
    # print(quad(integrand, by, 1.0)[0], I3(1.0) - I3(by))

    # num_approx = quad(integrand, 0, 1)[0]
    # print(num_approx, I1(ay) - I1(0.0) + I2(by) - I2(ay) + I3(1.0) - I3(by))

    return I1(ay) - I1(0.0) + I2(by) - I2(ay) + I3(1.0) - I3(by)


def true_integral_dux_dx_dugx_dx(ax, bx):
    dx = bx - ax

    def I(x):
        # I = int (1 - x**2) * (exp(2*x) - 1) dx
        #   = exp(2*x) * (1/4 + x/2 - x**2/2) + (x**3/3 - x)
        exp_part = np.exp(2 * x) * (0.25 + 0.5 * x - 0.5 * x**2)
        pol_part = x**3 / 3 - x
        return (exp_part + pol_part) / dx

    # from scipy.integrate import quad

    # def integrand(x):
    #     return true_f_dux_dx(x) * true_g_dux_dx(x, ax, bx)

    # print(quad(integrand, 0.0, 1.0)[0], I(bx) - I(ax))

    return I(bx) - I(ax)


def true_integral_duy_dy_dugy_dy(ay, by):
    dy = by - ay

    def I(y):
        # I = int 1 / dy * sin(pi * y) dy
        #   = - 1 / (dy * pi) * cos(pi * y)
        return -np.cos(np.pi * y) / (dy * np.pi)

    # from scipy.integrate import quad

    # def integrand(y):
    #     return true_f_duy_dy(y) * true_g_duy_dy(y, ay, by)

    # print(quad(integrand, 0.0, 1.0)[0], I(by) - I(ay))

    return I(by) - I(ay)
