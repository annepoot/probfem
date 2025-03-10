import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime

from probability.sampling import MCMCRunner
from experiments.inverse.three_point_hole.props import get_rwm_fem_target


def linear_tempering(i):
    if i > n_burn:
        return 1.0
    else:
        return i / n_burn


n_burn = 10000
n_sample = 20000
tempering = linear_tempering

std_corruption_range = [1e-3, 1e-4, 1e-5]
sigma_e_range = std_corruption_range
h_range = [0.2, 0.1, 0.05]
h_meas_range = [0.2, 0.5, 1.0]

write_output = True

if write_output:
    fname = "noise-study.csv"
    file = open(fname, "w")

    current_time = datetime.now().strftime("%Y/%d/%m, %H:%M:%S")
    file.write("author = Anne Poot\n")
    file.write(f"date, time = {current_time}\n")
    file.write(f"n_burn = {n_burn}\n")
    file.write(f"n_sample = {n_sample}\n")
    file.write(f"tempering = {tempering}\n")
    file.write(f"h = {h_range}\n")
    file.write(f"h_meas = fixed at {h_meas_range}\n")
    file.write(f"std_corruption = {std_corruption_range}\n")
    file.write(f"sigma_e = {sigma_e_range}\n")
    file.close()

for h in h_range:
    for h_meas in h_meas_range:
        for std_corruption, sigma_e in zip(std_corruption_range, sigma_e_range):
            target = get_rwm_fem_target(
                h=h,
                h_meas=h_meas,
                std_corruption=std_corruption,
                sigma_e=sigma_e,
            )

            proposal = deepcopy(target.prior)
            for dist in proposal.distributions:
                dist.update_width(0.1 * dist.calc_width())
            x_prop = proposal.distributions[0]
            y_prop = proposal.distributions[1]
            x_prop.update_width(y_prop.calc_width())
            start_value = np.array([1.0, 0.4, 0.4, np.pi / 6, 0.25])
            mcmc = MCMCRunner(
                target=target,
                proposal=proposal,
                n_sample=n_sample,
                n_burn=n_burn,
                start_value=start_value,
                seed=0,
                tempering=tempering,
                recompute_logpdf=False,
                return_info=True,
            )

            samples, info = mcmc()

            if write_output:
                df = pd.DataFrame(samples, columns=["x", "y", "a", "theta", "r_rel"])

                for header, data in info.items():
                    df[header] = data

                df["sample"] = df.index
                df["h"] = h
                df["h_meas"] = h_meas
                df["r"] = df["r_rel"] * df["a"]
                df["std_corruption"] = std_corruption
                df["sigma_e"] = sigma_e

                write_header = (
                    (h == h_range[0])
                    and (h_meas == h_meas_range[0])
                    and (sigma_e == sigma_e_range[0])
                )
                df.to_csv(fname, mode="a", header=write_header, index=False)
