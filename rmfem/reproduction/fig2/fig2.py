import numpy as np
import matplotlib.pyplot as plt

from myjive.app import main
from myjivex import declare_all
from myjive.util.proputils import split_off_type

from rmfem.rmfemrunner import RMFEMRunner
from fig2_props import fig2_props


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


for p in [1, 2]:
    for N in [5, 10, 20]:
        mesher_lin(1, N, fname="fig2.mesh")

        props_ref = fig2_props["inner"]
        inner_type, inner_kws = split_off_type(props_ref)
        assert inner_type is main.jive
        globdat_ref = main.jive(inner_kws, extra_declares=[declare_all])
        x_ref = globdat_ref["nodeSet"].get_coords().flatten()
        u_ref = globdat_ref["state0"]

        fig2_props["p"] = p

        rmfem = RMFEMRunner(**fig2_props)
        samples = rmfem()

        x_exact = np.linspace(0, 1, 100)
        u_exact = np.sin(x_exact * 2 * np.pi)

        plt.figure()
        for sample in samples:
            x_fem = sample["nodeSet"].get_coords().flatten()
            u_fem = sample["state0"]
            plt.plot(x_fem, u_fem, color="grey", alpha=0.3)
        plt.plot(x_exact, u_exact, color="black", linewidth=1)
        plt.plot(x_ref, u_ref)
        plt.title("Figure 2")
        plt.xlabel("x")
        plt.ylabel("Solution")
        plt.show()
