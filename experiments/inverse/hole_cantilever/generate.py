import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime

from fem.meshing.readwrite import write_mesh
from probability.sampling import MCMCRunner
from experiments.inverse.hole_cantilever.props import (
    get_rwm_fem_target,
    get_rwm_bfem_target,
    get_rwm_rmfem_target,
)
from experiments.inverse.hole_cantilever.meshing import create_mesh


n_burn = 10000
n_sample = 20000
std_corruption_range = [1e-3, 1e-4, 1e-5]
h_range = [0.2, 0.1, 0.05]
h_meas = 1.0

for fem_type in ["bfem"]:
    fname = "samples-{}.csv".format(fem_type)

    file = open(fname, "w")

    current_time = datetime.now().strftime("%Y/%d/%m, %H:%M:%S")
    file.write("author = Anne Poot\n")
    file.write(f"date, time = {current_time}\n")
    file.write(f"n_sample = {n_sample}\n")
    file.write(f"h = {h_range}\n")
    file.write(f"h_meas = fixed at {h_meas}\n")
    file.write(f"std_corruption = {std_corruption_range}\n")

    if fem_type == "fem":
        sigma_e_range = std_corruption_range
        recompute_logpdf = False
        file.write(f"sigma_e = {sigma_e_range}\n")

    elif fem_type == "bfem":
        sigma_e_range = std_corruption_range
        recompute_logpdf = False
        file.write(f"sigma_e = {sigma_e_range}\n")

    elif fem_type == "rmfem":
        sigma_e_range = std_corruption_range
        n_pseudomarginal = 10
        recompute_logpdf = True
        file.write(f"sigma_e = {sigma_e_range}\n")
        file.write(f"n_pseudomarginal = {n_pseudomarginal}\n")

    file.close()

    for std_corruption, sigma_e in zip(std_corruption_range, sigma_e_range):
        for i, h in enumerate(h_range):
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
                    scale=9.105562473643324e-07,  # f_c.T @ u_c / n_c
                    rescale=False,
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
            else:
                raise ValueError

            proposal = deepcopy(target.prior)
            for dist in proposal.distributions:
                dist.update_width(0.1 * dist.calc_width())
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

            df = pd.DataFrame(samples, columns=["x", "y", "a", "theta", "r_rel"])

            df["sample"] = df.index
            df["h"] = h
            df["r"] = df["r_rel"] * df["a"]
            df["std_corruption"] = std_corruption

            if fem_type == "fem":
                df["sigma_e"] = sigma_e
            elif fem_type == "rmfem":
                df["sigma_e"] = sigma_e
                df["n_pseudomarginal"] = n_pseudomarginal
            else:
                raise ValueError

            write_header = (h == h_range[0]) and (sigma_e == sigma_e_range[0])
            df.to_csv(fname, mode="a", header=write_header, index=False)

_, elems = create_mesh(
    h=0.1,
    L=4,
    H=1,
    x=1,
    y=0.4,
    a=0.4,
    theta=np.pi / 6,
    r_rel=0.25,
    h_meas=1.0,
)[0]

write_mesh(elems, "cantilever.msh")
