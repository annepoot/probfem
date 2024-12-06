import pandas as pd
from copy import deepcopy

from props.rwm_fem_hyper_props import get_rwm_fem_hyper_target
from props.rwm_statfem_hyper_props import get_rwm_statfem_hyper_target
from props.rwm_bfem_hyper_props import get_rwm_bfem_hyper_target
from probability.sampling import MCMCRunner

for fem_type in ["fem", "bfem", "statfem"]:
    for n_elem in [10, 20, 40]:
        std_corruption = 1e-5

        if fem_type == "fem":
            target = get_rwm_fem_hyper_target(
                n_elem=n_elem, std_corruption=std_corruption, n_rep_obs=1
            )
        elif fem_type == "bfem":
            target = get_rwm_bfem_hyper_target(
                n_elem=n_elem,
                std_corruption=std_corruption,
                n_rep_obs=1,
                sigma_e=std_corruption,
            )
        elif fem_type == "statfem":
            target = get_rwm_statfem_hyper_target(
                n_elem=n_elem,
                std_corruption=std_corruption,
                n_rep_obs=1,
                sigma_e=std_corruption,
            )
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

        fname = "samples-{}-hyper.csv".format(fem_type)

        if n_elem == 10:
            df.to_csv(fname, mode="w", header="column_names", index=False)
        else:
            df.to_csv(fname, mode="a", header=False, index=False)
