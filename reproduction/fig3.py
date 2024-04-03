import numpy as np
import matplotlib.pyplot as plt

from myjive.app import main
import myjive.util.proputils as pu
from declare import declare_all as declareloc
from myjivex import declare_all as declarex


def mesher_lin(L, n, fname="2nodebar"):
    dx = L / n
    if not "." in fname:
        fname += ".mesh"

    with open(fname, "w") as fmesh:
        fmesh.write("nodes (ID, x, [y], [z])\n")
        for i in range(n + 1):
            fmesh.write("%d %f\n" % (i, i * dx))
        fmesh.write("elements (node#1, node#2, [node#3, ...])\n")
        for i in range(n):
            fmesh.write("%d %d\n" % (i, i + 1))


props = pu.parse_file("fig3.pro")

extra_declares = [declarex, declareloc]
globdat = main.jive(props, extra_declares=extra_declares)


def get_x_h(globdat):
    x_h = np.zeros(len(globdat["state0"]))

    nset = globdat["nodeSet"]

    for i, node in enumerate(nset):
        x_h[i] = node.get_coords()[0]

    return x_h


def get_ielem(x, globdat):
    x_h = get_x_h(globdat)

    for i in range(len(x_h)):
        if x >= x_h[i] and x < x_h[i + 1]:
            return i


def u_exact(x):
    a = 15
    b = 50
    return x**3 * np.sin(a * np.pi * x) * np.exp(-b * (x - 0.5) ** 2)


def eps_exact(x):
    a = 15
    b = 50
    return (
        3 * x**2 * np.sin(a * np.pi * x) * np.exp(-b * (x - 0.5) ** 2)
        + x**3 * a * np.pi * np.cos(a * np.pi * x) * np.exp(-b * (x - 0.5) ** 2)
        + x**3
        * np.sin(a * np.pi * x)
        * (-2 * b * (x - 0.5) * np.exp(-b * (x - 0.5) ** 2))
    )


def u_fem(x, globdat):
    state0 = globdat["state0"]
    x_h = get_x_h(globdat)
    ielem = get_ielem(x, globdat)
    xl = x_h[ielem]
    xr = x_h[ielem + 1]

    return state0[ielem] + (x - xl) * (state0[ielem + 1] - state0[ielem]) / (xr - xl)


def eps_fem(x, globdat):
    state0 = globdat["state0"]
    x_h = get_x_h(globdat)
    ielem = get_ielem(x, globdat)
    xl = x_h[ielem]
    xr = x_h[ielem + 1]

    return (state0[ielem + 1] - state0[ielem]) / (xr - xl)


u_exact = np.vectorize(u_exact)
eps_exact = np.vectorize(eps_exact)
u_fem = np.vectorize(u_fem, excluded=[1])
eps_fem = np.vectorize(eps_fem, excluded=[1])


def error_u(x, globdat):
    u = u_exact(x)
    u_h = u_fem(x, globdat)

    return u - u_h


def error_eps(x, globdat):
    eps = eps_exact(x)
    eps_h = eps_fem(x, globdat)

    return eps - eps_h


from scipy.integrate import quad


def error_ui(ielem, globdat):
    def func(x):
        return np.sqrt(error_u(x, globdat) ** 2)

    x_h = get_x_h(globdat)
    xl = x_h[ielem]
    xr = x_h[ielem + 1]

    result = quad(func, xl, xr)

    return result[0]


def error_epsi(ielem, globdat):
    def func(x):
        return np.sqrt(error_eps(x, globdat) ** 2)

    x_h = get_x_h(globdat)
    xl = x_h[ielem]
    xr = x_h[ielem + 1]

    result = quad(func, xl, xr)

    return result[0]


# xf = np.linspace(0.0001, 0.9999, 1000)

# u = u_exact(xf)
# u_h = u_fem(xf, globdat)
# eps = eps_exact(xf)
# eps_h = eps_fem(xf, globdat)

# fig, ax = plt.subplots()
# ax.plot(xf, u)
# ax.plot(xf, u_h)
# for pglobdat in globdat["perturbedSolves"]:
#     u_ph = u_fem(xf, pglobdat)
#     ax.plot(xf, u_ph, color="gray", alpha = 0.3)
# plt.show()

# fig, ax = plt.subplots()
# ax.plot(xf, eps)
# ax.plot(xf, eps_h)
# for pglobdat in globdat["perturbedSolves"]:
#     eps_ph = eps_fem(xf, pglobdat)
#     ax.plot(xf, eps_ph, color="gray", alpha = 0.3)
# plt.show()

ielems = np.arange(1, len(globdat["state0"]) - 2)
u_error = np.zeros_like(ielems, dtype=float)
eps_error = np.zeros_like(ielems, dtype=float)
for i, ielem in enumerate(ielems):
    u_error[i] = error_ui(ielem, globdat)
    eps_error[i] = error_epsi(ielem, globdat)

nn = len(globdat["state0"])
ne = nn - 1

fig, ax = plt.subplots()
# ax.step(np.linspace(0, 1, ne)[1:-1], u_error)
ax.step(np.linspace(0, 1, ne)[1:-1], eps_error)
ax.step(np.linspace(0, 1, ne), globdat["eta"])
ax.set_yscale("log")
ax.set_ylim((1e-10, 1e0))
plt.show()

x = np.linspace(0, 1, ne, endpoint=False) + 1 / 2 / ne

fig, ax = plt.subplots()
# ax.plot(x[1:-1], u_error)
ax.plot(x[1:-1], eps_error)
ax.plot(x, globdat["eta"])
ax.set_yscale("log")
ax.set_ylim((1e-10, 1e0))
plt.show()
