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


nn = len(globdat["state0"])
ne = nn - 1

x = np.linspace(0, 1, ne, endpoint=False) + 1 / 2 / ne

u_error = globdat["tables"]["error"]["solution"]
eps_error = globdat["tables"]["error"]["strain"]
eta1 = globdat["tables"]["error"]["eta1"]
eta2 = globdat["tables"]["error"]["eta2"]

fig, ax = plt.subplots()
ax.plot(x, eps_error, color="k", label=r"$\|u' - u_h'\|_{L^2(K)}$")
ax.plot(x, eta1, label=r"$\eta_{K,1}$")
ax.plot(x, eta2, label=r"$\eta_{K,2}$")
ax.set_yscale("log")
ax.set_ylim((1e-10, 1e0))
ax.legend()
plt.savefig(fname="fig3_midproof-True", dpi=600)
plt.show()
