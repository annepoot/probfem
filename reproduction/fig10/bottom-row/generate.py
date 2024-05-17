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


extra_declares = [declarex, declareloc]
props = pu.parse_file("1d-inv-kl4.pro")
nsample = props["rmfem"]["nsample"]
files, keys = get_files_and_keys(10, 1e-8, nsample)
outputprops = {
    "type": "Output",
    "files": files,
    "keys": keys,
    "overwrite": True,
}
props["output"] = outputprops

for N in [10, 20, 40]:
    mesher_lin(1, N)
    for noise in [1e-8, 1e-10, 1e-12, 1e-14, 1e-16]:
        files, keys = get_files_and_keys(N, noise, nsample)
        props["model"]["obs"]["noise"]["cov"] = noise
        props["model"]["obs"]["measurement"]["corruption"]["cov"] = noise
        props["output"]["files"] = files
        props["output"]["keys"] = keys

        globdat = main.jive(props, extra_declares=extra_declares)

mesher_lin(1, 10)
