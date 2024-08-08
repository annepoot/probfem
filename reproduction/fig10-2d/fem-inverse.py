from myjive.app import main
import myjive.util.proputils as pu
from declare import declare_all as declareloc
from myjivex import declare_all as declarex


def get_file_names(mesh, noise):
    file1 = "output/fem-inverse/mcmc_xi_mesh-{}_noise-{}.csv".format(mesh, noise)
    file2 = "output/fem-inverse/mcmc_state0_mesh-{}_noise-{}.csv".format(mesh, noise)
    file3 = "output/fem-inverse/mcmc_stiffness_mesh-{}_noise-{}.csv".format(mesh, noise)
    return [file1, file2, file3]


extra_declares = [declarex, declareloc]
props = pu.parse_file("2d-fem-inv.pro")

outputprops = {
    "type": "Output",
    "files": get_file_names("bar_r0", 1e-08),
    "keys": ["mcmc.variables", "mcmc.state0", "mcmc.tables.stiffness"],
    "overwrite": True,
}
props["output"] = outputprops

for mesh in ["bar_r0", "bar_r1", "bar_r2"]:
    for noise in [1e-08, 1e-10, 1e-12]:
        props["init"]["mesh"]["file"] = "meshes/" + mesh + ".msh"
        props["model"]["obs"]["noise"]["cov"] = noise
        props["model"]["obs"]["measurement"]["corruption"]["cov"] = noise
        props["output"]["files"] = get_file_names(mesh, noise)

        globdat = main.jive(props, extra_declares=extra_declares)
