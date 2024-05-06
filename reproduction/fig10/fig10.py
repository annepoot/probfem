import numpy as np
import matplotlib.pyplot as plt

from myjive.app import main
import myjive.util.proputils as pu
from declare import declare_all as declareloc
from myjivex import declare_all as declarex


def mesher_lin(L, n, fname="fig10"):
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


props = pu.parse_file("fig10.pro")
extra_declares = [declarex, declareloc]
globdat = main.jive(props, extra_declares=extra_declares)

variables = globdat["mcmc"]["variables"]
solutions = globdat["mcmc"]["state0"]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.scatter(variables[:, 0], variables[:, 1])
ax2.scatter(variables[:, 0], variables[:, 2])
ax3.scatter(variables[:, 0], variables[:, 3])
ax4.scatter(variables[:, 1], variables[:, 2])
plt.show()


def get_kappa(x, xi):
    xi_1, xi_2, xi_3, xi_4 = xi[0], xi[1], xi[2], xi[3]
    exp, sqrt, pi, sin = np.exp, np.sqrt, np.pi, np.sin
    kappa = exp(
        sqrt(2)
        * (
            xi_1 / (pi) * sin(pi * x)
            + xi_2 / (2 * pi) * sin(2 * pi * x)
            + xi_3 / (3 * pi) * sin(3 * pi * x)
            + xi_4 / (4 * pi) * sin(4 * pi * x)
        )
    )
    return kappa


x = np.linspace(0, 1, 1000)
xi_true = [1.0, 1.0, 0.25, 0.25]
kappa_true = get_kappa(x, xi_true)

fig, ax = plt.subplots()
for xi in variables[500::10]:
    kappa = get_kappa(x, xi)
    ax.plot(x, kappa, color="gray", linewidth=0.2)
ax.plot(x, kappa_true, color="black")
ax.set_xlim((0, 1))
ax.set_ylim((0.8, 2.0))
plt.show()
