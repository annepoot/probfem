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


def get_file_names(N, noise):
    file1 = "output/mcmc_xi_N-{}_noise-{}.csv".format(N, noise)
    file2 = "output/mcmc_state0_N-{}_noise-{}.csv".format(N, noise)
    file3 = "output/mcmc_stiffness_N-{}_noise-{}.csv".format(N, noise)
    return [file1, file2, file3]


extra_declares = [declarex, declareloc]
props = pu.parse_file("1d-inv-kl6.pro")
outputprops = {
    "type": "Output",
    "files": get_file_names(10, 1e-8),
    "keys": ["mcmc.variables", "mcmc.state0", "mcmc.tables.stiffness"],
    "overwrite": True,
}
props["output"] = outputprops

for N in [10, 20, 40]:
    mesher_lin(1, N)
    for noise in [1e-8, 1e-10, 1e-12, 1e-14, 1e-16]:
        props["model"]["obs"]["noise"]["cov"] = noise
        props["model"]["obs"]["measurement"]["corruption"]["cov"] = noise
        props["output"]["files"] = get_file_names(N, noise)

        globdat = main.jive(props, extra_declares=extra_declares)

mesher_lin(1, 10)
