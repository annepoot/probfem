import pandas as pd
from copy import deepcopy
from datetime import datetime

from probability.sampling import MCMCRunner
from props.rwm_fem_props import get_rwm_fem_target
from props.rwm_statfem_props import get_rwm_statfem_target
from props.rwm_rmfem_props import get_rwm_rmfem_target
from props.rwm_bfem_props import get_rwm_bfem_target

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

for fem_type in ["statfem"]:
    fname = "samples-{}.csv".format(fem_type)
    current_time = datetime.now().strftime("%Y/%d/%m, %H:%M:%S")
    n_sample = 50000
    std_corruption = 1e-5
    n_elem_range = [10, 20, 40]

    file = open(fname, "w")

    current_time = datetime.now().strftime("%Y/%d/%m, %H:%M:%S")
    file.write("author = Anne Poot\n")
    file.write(f"date, time = {current_time}\n")
    file.write(f"n_sample = {n_sample}\n")
    file.write(f"n_elem = {n_elem_range}\n")
    file.write(f"std_corruption = fixed at {std_corruption}\n")

    if fem_type == "fem":
        sigma_e = std_corruption
        recompute_logpdf = False
        file.write(f"sigma_e = fixed at {sigma_e}\n")

    elif fem_type == "bfem":
        scale = 0.001044860592586493  # f_c.T @ u_c / n_c
        rescale = False
        sigma_e = std_corruption
        recompute_logpdf = True
        file.write(f"scale = fixed at {scale}\n")
        file.write(f"rescale = {rescale}\n")
        file.write(f"sigma_e = fixed at {sigma_e}\n")

    elif fem_type == "rmfem":
        sigma_e = std_corruption
        n_pseudomarginal = 10
        recompute_logpdf = True
        file.write(f"sigma_e = fixed at {sigma_e}\n")
        file.write(f"n_pseudomarginal = {n_pseudomarginal}\n")

    elif fem_type == "statfem":
        rho_range = [statfem_hparams[n_elem]["rho"] for n_elem in n_elem_range]
        l_d_range = [statfem_hparams[n_elem]["l_d"] for n_elem in n_elem_range]
        sigma_d_range = [statfem_hparams[n_elem]["sigma_d"] for n_elem in n_elem_range]
        sigma_e_range = [statfem_hparams[n_elem]["sigma_e"] for n_elem in n_elem_range]
        file.write(f"rho = {rho_range}\n")
        file.write(f"l_d = {l_d_range}\n")
        file.write(f"sigma_d = {sigma_d_range}\n")
        file.write(f"sigma_e = {sigma_e_range}\n")

    file.close()

    for i, n_elem in enumerate(n_elem_range):

        if fem_type == "fem":
            sigma_e = std_corruption
            target = get_rwm_fem_target(
                n_elem=n_elem,
                std_corruption=std_corruption,
                sigma_e=sigma_e,
                n_rep_obs=1,
            )
            recompute_logpdf = False

        elif fem_type == "bfem":
            target = get_rwm_bfem_target(
                n_elem=n_elem,
                std_corruption=std_corruption,
                scale=scale,  # f_c.T @ u_c / n_c
                rescale=rescale,
                sigma_e=sigma_e,
                n_rep_obs=1,
            )
            recompute_logpdf = False

        elif fem_type == "rmfem":
            target = get_rwm_rmfem_target(
                n_elem=n_elem,
                std_corruption=std_corruption,
                sigma_e=sigma_e,
                n_rep_obs=1,
                n_pseudomarginal=n_pseudomarginal,
            )
            recompute_logpdf = True

        elif fem_type == "statfem":
            rho = rho_range[i]
            l_d = l_d_range[i]
            sigma_d = sigma_d_range[i]
            sigma_e = sigma_e_range[i]
            target = get_rwm_statfem_target(
                n_elem=n_elem,
                std_corruption=std_corruption,
                rho=rho,
                l_d=l_d,
                sigma_d=sigma_d,
                sigma_e=sigma_e,
                n_rep_obs=1,
            )
            recompute_logpdf = False
        else:
            raise ValueError

        proposal = deepcopy(target.prior)
        start_value = target.prior.calc_mean()
        mcmc = MCMCRunner(
            target=target,
            proposal=proposal,
            n_sample=n_sample,
            start_value=start_value,
            seed=0,
            recompute_logpdf=recompute_logpdf,
        )
        samples = mcmc()

        df = pd.DataFrame(samples, columns=["xi_1", "xi_2", "xi_3", "xi_4"])

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
