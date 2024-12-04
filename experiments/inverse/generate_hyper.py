import numpy as np
import pandas as pd

from props.rwm_fem_hyper_props import get_rwm_fem_hyper_props
from props.rwm_statfem_hyper_props import get_rwm_statfem_hyper_props
from probability.sampling import MCMCRunner


def mesher_lin(L, n, fname="1d-lin"):
    dx = L / n
    if not "." in fname:
        fname += ".mesh"
    with open(fname, "w") as fmesh:
        fmesh.write("nodes (ID, x, [y], [z])\n")
        for i in range(n + 1):
            fmesh.write("%d %f\n" % (i, i * dx))
        fmesh.write("elements (node#1, node#2, [node#3, ...])\n")
        for i in range(n):
            fmesh.write("%d %d\n" % (i, i + 1))


for fem_type in ["fem", "statfem"]:
    for n_elem in [10, 20, 40]:
        mesher_lin(1, n_elem)

        std_corruption = 1e-5

        if fem_type == "fem":
            props = get_rwm_fem_hyper_props(std_corruption=std_corruption, n_rep_obs=1)
        elif fem_type == "statfem":
            props = get_rwm_statfem_hyper_props(
                std_corruption=std_corruption,
                n_rep_obs=1,
            )
        else:
            raise ValueError

        mcmc = MCMCRunner(**props)
        samples = mcmc()

        if fem_type == "fem":
            columns = ["xi_1", "xi_2", "xi_3", "xi_4", "log_sigma_e"]
        elif fem_type == "statfem":
            columns = [
                "xi_1",
                "xi_2",
                "xi_3",
                "xi_4",
                "log_rho",
                "log_l_d",
                "log_sigma_d",
                "log_sigma_e",
            ]
        else:
            raise ValueError

        df = pd.DataFrame(samples, columns=columns)

        df["sample"] = df.index
        df["n_elem"] = n_elem
        df["std_corruption"] = std_corruption

        if fem_type == "fem":
            df["sigma_e"] = np.exp(df["log_sigma_e"])
        elif fem_type == "statfem":
            df["rho"] = np.exp(df["log_rho"])
            df["l_d"] = np.exp(df["log_l_d"])
            df["sigma_d"] = np.exp(df["log_sigma_d"])
            df["sigma_e"] = np.exp(df["log_sigma_e"])
        else:
            raise ValueError

        fname = "samples-{}-hyper.csv".format(fem_type)

        if n_elem == 10:
            df.to_csv(fname, mode="w", header="column_names", index=False)
        else:
            df.to_csv(fname, mode="a", header=False, index=False)

mesher_lin(1, 10)
