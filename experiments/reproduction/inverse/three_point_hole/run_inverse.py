import os
import numpy as np
import pandas as pd
from datetime import datetime

from probability.sampling import RandomWalkMetropolisSampler, IndependenceSampler
from probability.multivariate import Gaussian, Mixture, EmpiricalMixture
from probability.reject import RejectConditional
from util.io import read_csv_from

from experiments.reproduction.inverse.three_point_hole.props import (
    get_rwm_fem_target,
    get_rwm_bfem_target,
    get_rwm_rmfem_target,
    get_rwm_statfem_target,
    rejection_func,
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
h_range = [0.20, 0.10, 0.05]
h_meas = 0.5

seed = "0-20"
write_output = True

for fem_type in ["fem", "bfem", "rmfem", "statfem"]:

    if write_output:
        fname = "samples-{}_seed-{}.csv".format(fem_type, seed)
        fname = os.path.join("output", fname)
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

        if isinstance(seed, int):
            rng = np.random.default_rng(seed)
            start_value = target.prior.latent.calc_mean()
            proposal = Gaussian(start_value, target.prior.latent.calc_cov())

            mcmc = RandomWalkMetropolisSampler(
                target=target,
                proposal=proposal,
                n_sample=n_sample,
                n_burn=n_burn,
                start_value=start_value,
                seed=rng,
                tempering=tempering,
                recompute_logpdf=recompute_logpdf,
                return_info=True,
            )

        elif isinstance(seed, str):
            start, end = [int(i) for i in seed.split("-")]
            ensemble = []

            for i in range(start, end):
                fname_df = "samples-{}_seed-{}.csv".format(fem_type, i)
                fname_df = os.path.join("output", fname_df)

                df = read_csv_from(fname_df, "x,y,a,theta,r_rel")
                df = df[df["sample"] >= n_burn]
                df = df[abs(df["h"] - h) < 1e-8]
                df = df[abs(df["std_corruption"] - std_corruption) < 1e-8]
                df["theta"] = np.fmod(df["theta"], 0.5 * np.pi)

                if fem_type == "statfem":
                    columns = [
                        "x",
                        "y",
                        "a",
                        "theta",
                        "r_rel",
                        "log_rho",
                        "log_l_d",
                        "log_sigma_d",
                    ]
                else:
                    columns = ["x", "y", "a", "theta", "r_rel"]

                df = df[columns]

                widths = np.max(df, axis=0) - np.min(df, axis=0)
                stds = 0.05 * widths
                covariance = np.diag(stds**2)
                ensemble.append(EmpiricalMixture(df, covariance))

            mixture = Mixture(ensemble)
            proposal = RejectConditional(latent=mixture, reject_if=rejection_func)

            # skip burn-in phase
            rng = np.random.default_rng(end)
            start_value = proposal.calc_sample(rng)
            n_sample = n_sample - n_burn
            n_burn = 0
            tempering = None

            mcmc = IndependenceSampler(
                target=target,
                proposal=proposal,
                n_sample=n_sample - n_burn,
                n_burn=0,
                start_value=start_value,
                seed=rng,
                recompute_logpdf=recompute_logpdf,
                return_info=True,
            )
        else:
            assert False

        samples, info = mcmc()

        if write_output:
            if fem_type == "statfem":
                columns = [
                    "x",
                    "y",
                    "a",
                    "theta",
                    "r_rel",
                    "log_rho",
                    "log_l_d",
                    "log_sigma_d",
                ]
            else:
                columns = ["x", "y", "a", "theta", "r_rel"]

            df = pd.DataFrame(samples, columns=columns)

            if fem_type == "statfem":
                df["rho"] = np.exp(df["log_rho"])
                df["l_d"] = np.exp(df["log_l_d"])
                df["sigma_d"] = np.exp(df["log_sigma_d"])

            for header, data in info.items():
                df[header] = data

            df["sample"] = df.index
            df["h"] = h
            df["r"] = df["r_rel"] * df["a"]
            df["std_corruption"] = std_corruption
            df["sigma_e"] = sigma_e

            if fem_type == "rmfem":
                df["n_pseudomarginal"] = n_pseudomarginal

            write_header = h == h_range[0]
            df.to_csv(fname, mode="a", header=write_header, index=False)
