import os
import numpy as np
from copy import deepcopy
import pytest

from myjive.fem import XNodeSet, XElementSet

from probability.sampling import MCMCRunner
from experiments.reproduction.probnum25.props import (
    get_rwm_fem_target,
    get_rwm_rmfem_target,
    get_rwm_fem_2d_target,
    get_rwm_rmfem_2d_target,
)


cwd = os.getcwd()
rootdir = os.path.join(cwd[: cwd.rfind(os.path.sep + "probfem")], "probfem")
fig2_path = os.path.join(rootdir, "experiments", "reproduction", "rmfem", "fig2")
fig3_path = os.path.join(rootdir, "experiments", "reproduction", "rmfem", "fig3")

# some code at the start of each script to suppress matplotlib from showing figures
prefix = ""
prefix += "import matplotlib\n"
prefix += "import warnings\n"
prefix += 'matplotlib.use("agg")\n'
prefix += 'warnings.filterwarnings("ignore", message="Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.")\n'
prefix += 'warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive, and thus cannot be shown")\n'

# some code at the end of each script to suppress matplotlib from showing figures
suffix = ""
suffix += "import matplotlib.pyplot as plt\n"
suffix += "plt.close()\n"


def generate_mesh(n_elem):
    node_coords = np.linspace(0, 1, n_elem + 1).reshape((-1, 1))
    nodes = XNodeSet()
    nodes.add_nodes(node_coords)
    nodes.to_nodeset()

    elem_inodes = np.array([np.arange(0, n_elem), np.arange(1, n_elem + 1)]).T
    elem_sizes = np.full(n_elem, 2)

    elems = XElementSet(nodes)
    elems.add_elements(elem_inodes, elem_sizes)
    elems.to_elementset()

    return nodes, elems


def generate_mesh_2d(n_elem):
    assert n_elem % 10 == 0
    n_rep = n_elem // 10

    node_coords_x = np.tile(np.linspace(0, 1, n_elem + 1), n_rep + 1)
    node_coords_y = np.repeat(np.linspace(0, 0.1, n_rep + 1), n_elem + 1)
    node_coords = np.array([node_coords_x, node_coords_y]).T
    nodes = XNodeSet()
    nodes.add_nodes(node_coords)
    nodes.to_nodeset()

    inodes_0 = np.arange(0, (n_elem + 1) * (n_rep))
    inodes_0 = inodes_0.reshape((n_rep, -1))[:, :-1].flatten()
    inodes_1 = inodes_0 + 1
    inodes_2 = inodes_0 + n_elem + 2
    inodes_3 = inodes_0 + n_elem + 1
    elem_inodes = np.array([inodes_0, inodes_1, inodes_2, inodes_3]).T
    elem_sizes = np.full(n_elem * n_rep, 4)

    elems = XElementSet(nodes)
    elems.add_elements(elem_inodes, elem_sizes)
    elems.to_elementset()

    return nodes, elems


@pytest.mark.probnum25
@pytest.mark.reproduction
@pytest.mark.values
def test_probnum25_reproduction_values():
    ref_map = get_reference_value_map()

    std_corruption = 1e-5
    n_elem_range = [10]
    n_sample = 100
    n_burn = 10000

    for fem_type in ["fem", "fem-2d", "rmfem", "rmfem-omit", "rmfem-2d"]:

        if fem_type in ["fem", "fem-2d"]:
            sigma_e = std_corruption
            recompute_logpdf = False

        elif fem_type in ["rmfem", "rmfem-2d"]:
            sigma_e = std_corruption
            n_pseudomarginal = 10
            recompute_logpdf = True
            omit_nodes = False

        elif fem_type == "rmfem-omit":
            sigma_e = std_corruption
            n_pseudomarginal = 10
            recompute_logpdf = True
            omit_nodes = True

        for i, n_elem in enumerate(n_elem_range):
            print(fem_type, n_elem, "\n")
            if "2d" in fem_type:
                nodes, elems = generate_mesh_2d(n_elem)
            else:
                nodes, elems = generate_mesh(n_elem)

            if fem_type == "fem":
                target = get_rwm_fem_target(
                    elems=elems,
                    std_corruption=std_corruption,
                    sigma_e=sigma_e,
                )
            elif fem_type == "fem-2d":
                target = get_rwm_fem_2d_target(
                    elems=elems,
                    std_corruption=std_corruption,
                    sigma_e=sigma_e,
                )
            elif fem_type == "rmfem":
                target = get_rwm_rmfem_target(
                    elems=elems,
                    std_corruption=std_corruption,
                    sigma_e=sigma_e,
                    n_pseudomarginal=n_pseudomarginal,
                    omit_nodes=False,
                )
            elif fem_type == "rmfem-2d":
                target = get_rwm_rmfem_2d_target(
                    elems=elems,
                    std_corruption=std_corruption,
                    sigma_e=sigma_e,
                    n_pseudomarginal=n_pseudomarginal,
                    omit_nodes=omit_nodes,
                )
            elif fem_type == "rmfem-omit":
                target = get_rwm_rmfem_target(
                    elems=elems,
                    std_corruption=std_corruption,
                    sigma_e=sigma_e,
                    n_pseudomarginal=n_pseudomarginal,
                    omit_nodes=omit_nodes,
                )
            else:
                raise ValueError

            proposal = deepcopy(target.prior)
            start_value = target.prior.calc_mean()
            mcmc = MCMCRunner(
                target=target,
                proposal=proposal,
                n_sample=n_sample,
                n_burn=n_burn,
                start_value=start_value,
                seed=0,
                recompute_logpdf=recompute_logpdf,
                return_info=True,
            )
            samples, info = mcmc()

            last_sample = samples[-1]
            last_logpdf = info["loglikelihood"][-1]

            ref_sample = ref_map[fem_type][n_elem][:4]
            ref_logpdf = ref_map[fem_type][n_elem][4]

            assert np.allclose(last_sample, ref_sample)
            assert np.isclose(last_logpdf, ref_logpdf)


def get_reference_value_map():
    ref_map = {
        "fem": {
            10: [
                1.1211063209170158,
                1.5183433823358872,
                0.9405048930419607,
                0.19035708605625995,
                -14697.628185986305,
            ],
            20: [
                1.1211063209170158,
                1.5183433823358872,
                0.9405048930419607,
                0.19035708605625995,
                -9932.418275872551,
            ],
            40: [
                1.1211063209170158,
                1.5183433823358872,
                0.9405048930419607,
                0.19035708605625995,
                -8911.169973970542,
            ],
        },
        "fem-2d": {
            10: [
                1.1211063209170158,
                1.5183433823358872,
                0.9405048930419607,
                0.19035708605625995,
                -14697.628185986305,
            ],
            20: [
                1.1211063209170158,
                1.5183433823358872,
                0.9405048930419607,
                0.19035708605625995,
                -9932.418275872551,
            ],
            40: [
                1.1211063209170158,
                1.5183433823358872,
                0.9405048930419607,
                0.19035708605625995,
                -8911.169973970542,
            ],
        },
        "rmfem": {
            10: [
                0.9086238004592972,
                1.761407622678727,
                1.9772915087406777,
                -0.9947279381587222,
                -14548.744367690048,
            ],
            20: [
                0.9314765197808177,
                1.316372093819294,
                1.5437801794489037,
                -0.18506706568131348,
                -11155.529112057404,
            ],
            40: [
                1.1211063209170158,
                1.5183433823358872,
                0.9405048930419607,
                0.19035708605625995,
                -8953.253342860566,
            ],
        },
        "rmfem-omit": {
            10: [
                0.9868414946943715,
                1.7284894989476276,
                1.2799920189082787,
                -0.6483912053911843,
                -14799.326980227837,
            ],
            20: [
                1.1211063209170158,
                1.5183433823358872,
                0.9405048930419607,
                0.19035708605625995,
                -9875.672815024516,
            ],
            40: [
                1.1211063209170158,
                1.5183433823358872,
                0.9405048930419607,
                0.19035708605625995,
                -8955.39587252731,
            ],
        },
        "rmfem-2d": {
            10: [
                0.9314765197808177,
                1.316372093819294,
                1.5437801794489037,
                -0.18506706568131348,
                -6780.337958151614,
            ],
            20: [
                1.1211063209170158,
                1.5183433823358872,
                0.9405048930419607,
                0.19035708605625995,
                -9452.510572814313,
            ],
            40: [
                1.1211063209170158,
                1.5183433823358872,
                0.9405048930419607,
                0.19035708605625995,
                -8899.7102210988,
            ],
        },
    }

    return ref_map

    # for fem_type in ["fem", "fem-2d", "rmfem", "rmfem-omit", "rmfem-2d"]:
