from myjive.app import main
import myjive.util.proputils as pu
from declare import declare_all as declareloc
from myjivex import declare_all as declarex


def mesher_lin(L, n, fname="1d-lin"):
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


extra_declares = [declarex, declareloc]
props = pu.parse_file("1d-inv-kl4.pro")
nsample = props["rmfem"]["nsample"]

globdat = main.jive(props, extra_declares=extra_declares)

from rmplotmodule import RMPlotModule
from myjive.util.proputils import split_off_type

typ, rmplotprops = split_off_type(props["rmplot"])

rmplot = RMPlotModule("rmplot")
rmplot.configure(
    globdat,
    field="solution",
    comp="dx",
    plotType="node",
    figure={"title": "Perturbed solution", "xlabel": "x", "ylabel": "Solution"},
    exact={"color": "black", "linewidth": 1},
    perturbed={"color": "grey", "alpha": 0.3},
)
rmplot.init(globdat)
rmplot.run(globdat)
rmplot.shutdown(globdat)


import numpy as np
import matplotlib.pyplot as plt

xi = globdat["perturbedSolves"][0]["mcmc"]["variables"][-1]
xi_1, xi_2, xi_3, xi_4 = xi[0], xi[1], xi[2], xi[3]
sqrt, exp, sin, pi = np.sqrt, np.exp, np.sin, np.pi
x = np.linspace(0, 1, 1000)
E = exp(
    sqrt(2)
    * (
        xi_1 / (pi) * sin(pi * x)
        + xi_2 / (2 * pi) * sin(2 * pi * x)
        + xi_3 / (3 * pi) * sin(3 * pi * x)
        + xi_4 / (4 * pi) * sin(4 * pi * x)
    )
)

plt.figure()
plt.plot(x, E)
plt.show()

N_burn = 15000

u_1 = globdat["perturbedSolves"][0]["mcmc"]["state0"][N_burn:, 1]
u_2 = globdat["perturbedSolves"][0]["mcmc"]["state0"][N_burn:, 2]
u_3 = globdat["perturbedSolves"][0]["mcmc"]["state0"][N_burn:, 3]
u_4 = globdat["perturbedSolves"][0]["mcmc"]["state0"][N_burn:, 4]

plt.figure()
plt.scatter(u_1, u_2, color="C0")
plt.scatter([0.9510565162951536], [0.5877852522924731], color="k")
plt.show()


def get_files_and_keys(N, noise, nsample):
    files = []
    keys = []

    files.append("output/mcmc_xi_N-{}_noise-{}_mesh-ref.csv".format(N, noise))
    files.append("output/mcmc_state0_N-{}_noise-{}_mesh-ref.csv".format(N, noise))
    files.append("output/mcmc_stiffness_N-{}_noise-{}_mesh-ref.csv".format(N, noise))

    keys.append("mcmc.variables.0")
    keys.append("mcmc.state0.0")
    keys.append("mcmc.tables.stiffness..0")

    for sample in range(nsample):
        files.append(
            "output/mcmc_xi_N-{}_noise-{}_mesh-{}.csv".format(N, noise, sample + 1)
        )
        files.append(
            "output/mcmc_state0_N-{}_noise-{}_mesh-{}.csv".format(N, noise, sample + 1)
        )
        files.append(
            "output/mcmc_stiffness_N-{}_noise-{}_mesh-{}.csv".format(
                N, noise, sample + 1
            )
        )

        keys.append("perturbedSolves.{}.mcmc.variables".format(sample))
        keys.append("perturbedSolves.{}.mcmc.state0".format(sample))
        keys.append("perturbedSolves.{}.mcmc.tables.stiffness".format(sample))

    return files, keys


files, keys = get_files_and_keys(5, 1e-20, nsample)
outputprops = {
    "type": "Output",
    "files": files,
    "keys": keys,
    "overwrite": True,
}
props["output"] = outputprops

from myjive.app.outputmodule import OutputModule
from myjive.util.proputils import split_off_type

output = OutputModule("output")
typ, outputprops = split_off_type(outputprops)
outputprops
outputprops.keys()
output.configure(globdat, **outputprops)
output.init(globdat)
output.run(globdat)

from writemeshmodule import WriteMeshModule

writemesh = WriteMeshModule("writemesh")

writemesh.WRITEMESH(globdat, "mesh-ref.mesh", "manual")
for i, pglobdat in enumerate(globdat["perturbedSolves"], 1):
    writemesh.WRITEMESH(pglobdat, "mesh-{}.mesh".format(i), "manual")
