import os
import numpy as np
import pandas as pd
from datetime import datetime

from myjive.fem import XNodeSet, XElementSet
from probability.sampling import MCMCRunner
from probability.multivariate import Gaussian
from experiments.reproduction.inverse.pullout_bar.props import (
    get_rwm_fem_target,
    get_rwm_bfem_target,
    get_rwm_rmfem_target,
    get_rwm_statfem_target,
)


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


def linear_tempering(i):
    if i > n_burn:
        return 1.0
    else:
        return i / n_burn


n_burn = 10000
n_sample = 20000
tempering = linear_tempering

std_corruption = 1e-3
n_elem_range = [1, 2, 4, 8, 16, 32, 64, 128]

write_output = True

for fem_type in ["fem", "bfem", "rmfem", "statfem"]:

    if write_output:
        fname = os.path.join("output", "samples-{}.csv".format(fem_type))
        file = open(fname, "w")

        current_time = datetime.now().strftime("%Y/%d/%m, %H:%M:%S")
        file.write("author = Anne Poot\n")
        file.write(f"date, time = {current_time}\n")
        file.write(f"n_burn = {n_burn}\n")
        file.write(f"n_sample = {n_sample}\n")
        file.write(f"tempering = {tempering}\n")
        file.write(f"n_elem = {n_elem_range}\n")
        file.write(f"std_corruption = fixed at {std_corruption}\n")

    if fem_type == "fem":
        sigma_e = std_corruption
        recompute_logpdf = False

        if write_output:
            file.write(f"sigma_e = fixed at {sigma_e}\n")

    elif fem_type == "bfem":
        scale = "mle"  # f_c.T @ u_c / n_c
        sigma_e = std_corruption
        recompute_logpdf = False

        if write_output:
            file.write(f"scale = {scale}\n")
            file.write(f"sigma_e = fixed at {sigma_e}\n")

    elif fem_type == "rmfem":
        sigma_e = std_corruption
        n_pseudomarginal = 100
        recompute_logpdf = True

        if write_output:
            file.write(f"sigma_e = fixed at {sigma_e}\n")
            file.write(f"n_pseudomarginal = {n_pseudomarginal}\n")

    elif fem_type == "statfem":
        sigma_e = std_corruption
        recompute_logpdf = False

        if write_output:
            file.write(f"sigma_e = fixed at {sigma_e}\n")

    if write_output:
        file.close()

    for i, n_elem in enumerate(n_elem_range):
        nodes, elems = generate_mesh(n_elem)

        if fem_type == "fem":
            target = get_rwm_fem_target(
                elems=elems,
                std_corruption=std_corruption,
                sigma_e=sigma_e,
            )
        elif fem_type == "bfem":
            ref_nodes, ref_elems = generate_mesh(2 * n_elem)
            target = get_rwm_bfem_target(
                obs_elems=elems,
                ref_elems=ref_elems,
                std_corruption=std_corruption,
                scale=scale,  # f_c.T @ u_c / n_c
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
        elif fem_type == "statfem":
            target = get_rwm_statfem_target(
                elems=elems,
                std_corruption=std_corruption,
                sigma_e=sigma_e,
            )
        else:
            raise ValueError

        start_value = target.prior.calc_mean()
        proposal = Gaussian(start_value, target.prior.calc_cov())
        mcmc = MCMCRunner(
            target=target,
            proposal=proposal,
            n_sample=n_sample,
            n_burn=n_burn,
            start_value=start_value,
            seed=0,
            tempering=tempering,
            recompute_logpdf=recompute_logpdf,
            return_info=True,
        )
        samples, info = mcmc()

        if write_output:
            if fem_type == "statfem":
                columns = ["log_E", "log_k", "log_rho", "log_l_d", "log_sigma_d"]
            else:
                columns = ["log_E", "log_k"]

            df = pd.DataFrame(samples, columns=columns)

            df["E"] = np.exp(df["log_E"])
            df["k"] = np.exp(df["log_k"])

            if fem_type == "statfem":
                df["rho"] = np.exp(df["log_rho"])
                df["l_d"] = np.exp(df["log_l_d"])
                df["sigma_d"] = np.exp(df["log_sigma_d"])

            for header, data in info.items():
                df[header] = data

            df["sample"] = df.index
            df["n_elem"] = n_elem
            df["std_corruption"] = std_corruption
            df["sigma_e"] = sigma_e

            if fem_type == "rmfem":
                df["n_pseudomarginal"] = n_pseudomarginal

            write_header = n_elem == n_elem_range[0]
            df.to_csv(fname, mode="a", header=write_header, index=False)
