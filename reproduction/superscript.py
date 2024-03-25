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


props = pu.parse_file("reproduction.pro")

extra_declares = [declarex, declareloc]
globdat = main.jive(props, extra_declares=extra_declares)

a = 15
b = 50
xc = np.linspace(0,1,len(globdat["state0"]))
xf = np.linspace(0,1,1000)
yf = xf**3 * np.sin(a*np.pi*xf) * np.exp(-b*(xf-0.5)**2)
plt.figure()
plt.plot(xc, globdat["state0"])
plt.plot(xf,yf, color="gray")
plt.show()
