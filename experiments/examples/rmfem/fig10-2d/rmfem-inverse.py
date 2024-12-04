from myjive.app import main
import myjive.util.proputils as pu
from rmfem import declare_all as declarermfem
from bayes import declare_all as declarebayes
from myjivex import declare_all as declarex


def get_files_and_keys(mesh, noise, nsample):
    files = []
    keys = []

    folder = "output/rmfem-inverse/"

    file1 = folder + "mcmc_xi_mesh-{}_noise-{}_mesh-ref.csv".format(mesh, noise)
    file2 = folder + "mcmc_state0_mesh-{}_noise-{}_mesh-ref.csv".format(mesh, noise)
    file3 = folder + "mcmc_stiffness_mesh-{}_noise-{}_mesh-ref.csv".format(mesh, noise)

    files.append(file1)
    files.append(file2)
    files.append(file3)

    keys.append("mcmc.variables.0")
    keys.append("mcmc.state0.0")
    keys.append("mcmc.tables.stiffness..0")

    for sample in range(nsample):
        file1 = folder + "mcmc_xi_mesh-{}_noise-{}_mesh-{}.csv".format(
            mesh, noise, sample + 1
        )
        file2 = folder + "mcmc_state0_mesh-{}_noise-{}_mesh-{}.csv".format(
            mesh, noise, sample + 1
        )
        file3 = folder + "mcmc_stiffness_mesh-{}_noise-{}_mesh-{}.csv".format(
            mesh, noise, sample + 1
        )

        files.append(file1)
        files.append(file2)
        files.append(file3)

        keys.append("perturbedSolves.{}.mcmc.variables".format(sample))
        keys.append("perturbedSolves.{}.mcmc.state0".format(sample))
        keys.append("perturbedSolves.{}.mcmc.tables.stiffness".format(sample))

    return files, keys


extra_declares = [declarex, declarermfem, declarebayes]
props = pu.parse_file("2d-rmfem-inv.pro")
nsample = props["rmfem"]["nsample"]
files, keys = get_files_and_keys("bar_r0", 1e-08, nsample)
outputprops = {
    "type": "Output",
    "files": files,
    "keys": keys,
    "overwrite": True,
}
props["output"] = outputprops

for mesh in ["bar_r0", "bar_r1", "bar_r2"]:
    for noise in [1e-08, 1e-10, 1e-12]:
        files, keys = get_files_and_keys(mesh, noise, nsample)
        props["init"]["mesh"]["file"] = "meshes/" + mesh + ".msh"
        props["model"]["obs"]["noise"]["cov"] = noise
        props["model"]["obs"]["measurement"]["corruption"]["cov"] = noise
        props["output"]["files"] = files
        props["output"]["keys"] = keys

        globdat = main.jive(props, extra_declares=extra_declares)
