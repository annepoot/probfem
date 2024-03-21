import matplotlib.pyplot as plt
import numpy as np
from myjive.app import main
import myjive.util.proputils as pu
from myjive.solver import Constrainer
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


props = pu.parse_file("tapered.pro")

extra_declares = [declarex, declareloc]
globdat = main.jive(props, extra_declares=extra_declares)

K = globdat["matrix0"]
u = globdat["state0"]
f = globdat["extForce"]
c = globdat["constraints"]

conman = Constrainer(c, K)
Kc = conman.get_output_matrix()
fc = conman.get_rhs(f)

xf = np.linspace(0, 1, len(u))

plt.figure()
plt.plot(xf, u, label="solution")
plt.legend()
plt.show()
