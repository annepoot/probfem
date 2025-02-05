import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime

from probability.sampling import MCMCRunner
from experiments.reproduction.probnum25.props import (
    get_rwm_fem_target,
    get_rwm_rmfem_target,
    get_rwm_fem_2d_target,
    get_rwm_rmfem_2d_target,
)

n_burn = 10000
n_sample = 20000
std_corruption = 1e-5
n_elem_range = [10, 20, 40]

from myjive.fem import XNodeSet, XElementSet


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


for fem_type in ["fem", "fem-2d", "rmfem", "rmfem-omit", "rmfem-2d"]:
    fname = "output/samples-{}.csv".format(fem_type)

    file = open(fname, "w")

    current_time = datetime.now().strftime("%Y/%d/%m, %H:%M:%S")
    file.write("author = Anne Poot\n")
    file.write(f"date, time = {current_time}\n")
    file.write(f"n_burn = {n_burn}\n")
    file.write(f"n_sample = {n_sample}\n")
    file.write(f"n_elem = {n_elem_range}\n")
    file.write(f"std_corruption = fixed at {std_corruption}\n")

    if fem_type in ["fem", "fem-2d"]:
        sigma_e = std_corruption
        recompute_logpdf = False
        file.write(f"sigma_e = fixed at {sigma_e}\n")

    elif fem_type in ["rmfem", "rmfem-2d"]:
        sigma_e = std_corruption
        n_pseudomarginal = 10
        recompute_logpdf = True
        omit_nodes = False
        file.write(f"sigma_e = fixed at {sigma_e}\n")
        file.write(f"n_pseudomarginal = {n_pseudomarginal}\n")
        file.write(f"omit_nodes = {omit_nodes}\n")

    elif fem_type == "rmfem-omit":
        sigma_e = std_corruption
        n_pseudomarginal = 10
        recompute_logpdf = True
        omit_nodes = True
        file.write(f"sigma_e = fixed at {sigma_e}\n")
        file.write(f"n_pseudomarginal = {n_pseudomarginal}\n")
        file.write(f"omit_nodes = {omit_nodes}\n")

    file.close()

    for i, n_elem in enumerate(n_elem_range):
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
        )
        samples = mcmc()

        df = pd.DataFrame(samples, columns=["xi_1", "xi_2", "xi_3", "xi_4"])

        df["sample"] = df.index
        df["n_elem"] = n_elem
        df["std_corruption"] = std_corruption

        if fem_type in ["fem", "fem-2d"]:
            df["sigma_e"] = sigma_e
        elif fem_type in ["rmfem", "rmfem-2d"]:
            df["sigma_e"] = sigma_e
            df["n_pseudomarginal"] = n_pseudomarginal
        elif fem_type == "rmfem-omit":
            df["sigma_e"] = sigma_e
            df["n_pseudomarginal"] = n_pseudomarginal
        else:
            raise ValueError

        write_header = n_elem == n_elem_range[0]
        df.to_csv(fname, mode="a", header=write_header, index=False)
