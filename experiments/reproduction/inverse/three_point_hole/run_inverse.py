import os
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime

from probability.univariate import Uniform, LogGaussian
from probability.sampling import MCMCRunner
from experiments.reproduction.inverse.three_point_hole.props import (
    get_rwm_fem_target,
    get_rwm_bfem_target,
    get_rwm_rmfem_target,
    get_rwm_statfem_target,
)


def linear_tempering(i):
    if i > n_burn:
        return 1.0
    else:
        return i / n_burn


n_burn = 10000
n_sample = 20000
tempering = linear_tempering

std_corruption = 1e-4
h_range = [0.2, 0.1, 0.05]
h_meas = 0.5

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
        file.write(f"h = {h_range}\n")
        file.write(f"h_meas = fixed at {h_meas}\n")
        file.write(f"std_corruption = {std_corruption}\n")

    if fem_type == "fem":
        sigma_e = std_corruption
        recompute_logpdf = False

        if write_output:
            file.write(f"sigma_e = {sigma_e}\n")

    elif fem_type == "bfem":
        sigma_e = std_corruption
        recompute_logpdf = False

        if write_output:
            file.write(f"sigma_e = {sigma_e}\n")

    elif fem_type == "rmfem":
        sigma_e = std_corruption
        n_pseudomarginal = 10
        recompute_logpdf = True

        if write_output:
            file.write(f"sigma_e = {sigma_e}\n")
            file.write(f"n_pseudomarginal = {n_pseudomarginal}\n")

    elif fem_type == "statfem":
        sigma_e = std_corruption
        recompute_logpdf = False

        if write_output:
            file.write(f"sigma_e = {sigma_e}\n")

    if write_output:
        file.close()

    for h in h_range:
        if fem_type == "fem":
            target = get_rwm_fem_target(
                h=h,
                h_meas=h_meas,
                std_corruption=std_corruption,
                sigma_e=sigma_e,
            )
        elif fem_type == "bfem":
            target = get_rwm_bfem_target(
                h=h,
                h_meas=h_meas,
                std_corruption=std_corruption,
                scale="mle",  # f_c.T @ u_c / n_c
                sigma_e=sigma_e,
            )
        elif fem_type == "rmfem":
            target = get_rwm_rmfem_target(
                h=h,
                h_meas=h_meas,
                std_corruption=std_corruption,
                sigma_e=sigma_e,
                n_pseudomarginal=n_pseudomarginal,
            )
        elif fem_type == "statfem":
            target = get_rwm_statfem_target(
                h=h,
                h_meas=h_meas,
                std_corruption=std_corruption,
                sigma_e=sigma_e,
            )
        else:
            raise ValueError

        proposal = deepcopy(target.prior)
        for dist in proposal.distributions:
            if isinstance(dist, Uniform):
                dist.update_width(0.1 * dist.calc_width())
            elif isinstance(dist, LogGaussian):
                dist.update_latent_std(0.1 * dist.calc_latent_std())
            else:
                assert False

        x_prop = proposal.distributions[0]
        y_prop = proposal.distributions[1]
        x_prop.update_width(y_prop.calc_width())
        start_value = proposal.calc_mean()
        start_value[:5] = np.array([1.0, 0.4, 0.4, np.pi / 6, 0.25])
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
                columns = ["x", "y", "a", "theta", "r_rel", "rho", "l_d", "sigma_d"]
            else:
                columns = ["x", "y", "a", "theta", "r_rel"]

            df = pd.DataFrame(samples, columns=columns)

            for header, data in info.items():
                df[header] = data

            df["sample"] = df.index
            df["h"] = h
            df["r"] = df["r_rel"] * df["a"]
            df["std_corruption"] = std_corruption

            if fem_type == "fem":
                df["sigma_e"] = sigma_e
            elif fem_type == "bfem":
                df["sigma_e"] = sigma_e
            elif fem_type == "rmfem":
                df["sigma_e"] = sigma_e
                df["n_pseudomarginal"] = n_pseudomarginal
            elif fem_type == "statfem":
                df["sigma_e"] = sigma_e
            else:
                raise ValueError

            write_header = h == h_range[0]
            df.to_csv(fname, mode="a", header=write_header, index=False)
