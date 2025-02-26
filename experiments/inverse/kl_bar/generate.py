import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime

from myjive.fem import XNodeSet, XElementSet
from probability.sampling import MCMCRunner
from experiments.inverse.kl_bar.props import (
    get_rwm_fem_target,
    get_rwm_statfem_target,
    get_rwm_rmfem_target,
    get_rwm_bfem_target,
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


statfem_hparams = {
    10: {
        "rho": 1.013586,
        "l_d": 0.178171,
        "sigma_d": 0.000063,
        "sigma_e": 0.000036,
    },
    20: {
        "rho": 0.995205,
        "l_d": 0.106763,
        "sigma_d": 0.000130,
        "sigma_e": 0.000073,
    },
    40: {
        "rho": 0.983613,
        "l_d": 0.793182,
        "sigma_d": 0.000773,
        "sigma_e": 0.000041,
    },
}


def linear_tempering(i):
    if i > n_burn:
        return 1.0
    else:
        return i / n_burn


n_burn = 10000
n_sample = 20000
tempering = linear_tempering

std_corruption = 1e-5
n_elem_range = [10, 20, 40]

write_output = True

for fem_type in ["fem", "bfem", "rmfem", "statfem"]:

    if write_output:
        fname = "samples-{}.csv".format(fem_type)
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
        n_pseudomarginal = 10
        recompute_logpdf = True

        if write_output:
            file.write(f"sigma_e = fixed at {sigma_e}\n")
            file.write(f"n_pseudomarginal = {n_pseudomarginal}\n")

    elif fem_type == "statfem":
        rho_range = [statfem_hparams[n_elem]["rho"] for n_elem in n_elem_range]
        l_d_range = [statfem_hparams[n_elem]["l_d"] for n_elem in n_elem_range]
        sigma_d_range = [statfem_hparams[n_elem]["sigma_d"] for n_elem in n_elem_range]
        sigma_e_range = [statfem_hparams[n_elem]["sigma_e"] for n_elem in n_elem_range]
        recompute_logpdf = False

        if write_output:
            file.write(f"rho = {rho_range}\n")
            file.write(f"l_d = {l_d_range}\n")
            file.write(f"sigma_d = {sigma_d_range}\n")
            file.write(f"sigma_e = {sigma_e_range}\n")

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
            ref_nodes, ref_elems = generate_mesh(80)
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
            rho = rho_range[i]
            l_d = l_d_range[i]
            sigma_d = sigma_d_range[i]
            sigma_e = sigma_e_range[i]
            target = get_rwm_statfem_target(
                elems=elems,
                std_corruption=std_corruption,
                rho=rho,
                l_d=l_d,
                sigma_d=sigma_d,
                sigma_e=sigma_e,
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
            tempering=tempering,
            recompute_logpdf=recompute_logpdf,
            return_info=True,
        )
        samples, info = mcmc()

        if write_output:
            df = pd.DataFrame(samples, columns=["xi_1", "xi_2", "xi_3", "xi_4"])

            for header, data in info.items():
                df[header] = data

            df["sample"] = df.index
            df["n_elem"] = n_elem
            df["std_corruption"] = std_corruption

            if fem_type == "fem":
                df["sigma_e"] = sigma_e
            elif fem_type == "bfem":
                df["sigma_e"] = sigma_e
            elif fem_type == "rmfem":
                df["sigma_e"] = sigma_e
                df["n_pseudomarginal"] = n_pseudomarginal
            elif fem_type == "statfem":
                df["std_corruption"] = std_corruption
                df["rho"] = rho
                df["l_d"] = l_d
                df["sigma_d"] = sigma_d
                df["sigma_e"] = sigma_e
            else:
                raise ValueError

            write_header = n_elem == n_elem_range[0]
            df.to_csv(fname, mode="a", header=write_header, index=False)
