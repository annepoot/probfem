import pandas as pd
from copy import deepcopy

from probability.sampling import MCMCRunner
from props.rwm_fem_props import get_rwm_fem_target
from props.rwm_statfem_props import get_rwm_statfem_target
from props.rwm_rmfem_props import get_rwm_rmfem_target
from props.rwm_bfem_props import get_rwm_bfem_target

statfem_hyperparams = {
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

for fem_type in ["fem", "bfem", "rmfem", "statfem"]:
    for n_elem in [10, 20, 40]:
        std_corruption = 1e-5

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
            sigma_e = std_corruption
            target = get_rwm_bfem_target(
                n_elem=n_elem,
                std_corruption=std_corruption,
                scale=0.001044860592586493,  # f_c.T @ u_c / n_c
                sigma_e=sigma_e,
                n_rep_obs=1,
            )
            recompute_logpdf = False
        elif fem_type == "rmfem":
            sigma_e = std_corruption
            n_pseudomarginal = 10
            target = get_rwm_rmfem_target(
                n_elem=n_elem,
                std_corruption=std_corruption,
                sigma_e=sigma_e,
                n_rep_obs=1,
                n_pseudomarginal=n_pseudomarginal,
            )
            recompute_logpdf = True
        elif fem_type == "statfem":
            rho = statfem_hyperparams[n_elem]["rho"]
            l_d = statfem_hyperparams[n_elem]["l_d"]
            sigma_d = statfem_hyperparams[n_elem]["sigma_d"]
            sigma_e = statfem_hyperparams[n_elem]["sigma_e"]
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
            n_sample=10000,
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
        elif fem_type == "statfem":
            df["std_corruption"] = std_corruption
            df["rho"] = rho
            df["l_d"] = l_d
            df["sigma_d"] = sigma_d
            df["sigma_e"] = sigma_e
        elif fem_type == "rmfem":
            df["sigma_e"] = sigma_e
            df["n_pseudomarginal"] = n_pseudomarginal
        elif fem_type == "bfem":
            df["sigma_e"] = sigma_e
        else:
            raise ValueError

        fname = "samples-{}.csv".format(fem_type)

        if n_elem == 10:
            df.to_csv(fname, mode="w", header="column_names", index=False)
        else:
            df.to_csv(fname, mode="a", header=False, index=False)
