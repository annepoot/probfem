import pandas as pd
from copy import deepcopy
from datetime import datetime

from props.rwm_fem_hyper_props import get_rwm_fem_hyper_target
from props.rwm_statfem_hyper_props import get_rwm_statfem_hyper_target
from props.rwm_bfem_hyper_props import get_rwm_bfem_hyper_target
from probability.sampling import MCMCRunner

for fem_type in ["fem", "bfem", "statfem"]:
    fname = "samples-{}-hyperprior.csv".format(fem_type)
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
        file.write("sigma_e = learned with log(prior) ~ N(0, log(1e1))\n")

    elif fem_type == "bfem":
        rescale = False
        sigma_e = std_corruption
        file.write("scale = learned with log(prior) ~ N(0, log(1e1))\n")
        file.write(f"rescale = {rescale}\n")
        file.write(f"sigma_e = fixed at {sigma_e}\n")

    elif fem_type == "statfem":
        sigma_e = std_corruption
        file.write("rho = learned with log(prior) ~ N(0, log(1e1))\n")
        file.write("l_d = learned with log(prior) ~ N(0, log(1e1))\n")
        file.write("sigma_d = learned with log(prior) ~ N(log(1e-4), log(1e1))\n")
        file.write(f"sigma_e = fixed at {sigma_e}\n")

    file.close()

    for n_elem in n_elem_range:

        if fem_type == "fem":
            target = get_rwm_fem_hyper_target(
                n_elem=n_elem, std_corruption=std_corruption, n_rep_obs=1
            )
        elif fem_type == "bfem":
            target = get_rwm_bfem_hyper_target(
                n_elem=n_elem,
                std_corruption=std_corruption,
                rescale=rescale,
                sigma_e=sigma_e,
                n_rep_obs=1,
            )
        elif fem_type == "statfem":
            target = get_rwm_statfem_hyper_target(
                n_elem=n_elem,
                std_corruption=std_corruption,
                sigma_e=sigma_e,
                n_rep_obs=1,
            )
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
        )
        samples = mcmc()

        if fem_type == "fem":
            columns = ["xi_1", "xi_2", "xi_3", "xi_4", "sigma_e"]
        elif fem_type == "statfem":
            columns = [
                "xi_1",
                "xi_2",
                "xi_3",
                "xi_4",
                "rho",
                "l_d",
                "sigma_d",
            ]
        elif fem_type == "bfem":
            columns = ["xi_1", "xi_2", "xi_3", "xi_4", "scale"]
        else:
            raise ValueError

        df = pd.DataFrame(samples, columns=columns)

        df["sample"] = df.index
        df["n_elem"] = n_elem
        df["std_corruption"] = std_corruption

        if fem_type == "fem":
            pass
        elif fem_type == "bfem":
            df["sigma_e"] = sigma_e
        elif fem_type == "statfem":
            df["sigma_e"] = sigma_e
        else:
            raise ValueError

        write_header = n_elem == n_elem_range[0]
        df.to_csv(fname, mode="a", header=write_header, index=False)
