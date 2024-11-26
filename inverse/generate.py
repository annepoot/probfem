import pandas as pd

from props.rwm_fem_props import get_rwm_fem_props
from props.rwm_statfem_props import get_rwm_statfem_props
from props.rwm_rmfem_props import get_rwm_rmfem_props
from sampling import MCMCRunner


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


for fem_type in ["fem", "statfem", "rmfem"]:
    for n_elem in [10, 20, 40]:
        mesher_lin(1, n_elem)

        std_corruption = 1e-5

        if fem_type == "fem":
            sigma_e = std_corruption
            props = get_rwm_fem_props(
                std_corruption=std_corruption, sigma_e=sigma_e, n_rep_obs=1
            )
        elif fem_type == "statfem":
            rho = 1.0
            l_d = 1.0
            sigma_d = 1e-8
            sigma_e = std_corruption
            props = get_rwm_statfem_props(
                std_corruption=std_corruption,
                rho=rho,
                l_d=l_d,
                sigma_d=sigma_d,
                sigma_e=sigma_e,
                n_rep_obs=1,
            )
        elif fem_type == "rmfem":
            sigma_e = std_corruption
            n_pseudomarginal = 10
            props = get_rwm_rmfem_props(
                std_corruption=std_corruption,
                sigma_e=sigma_e,
                n_rep_obs=1,
                n_pseudomarginal=n_pseudomarginal,
            )
        else:
            raise ValueError

        mcmc = MCMCRunner(**props)
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
        else:
            raise ValueError

        fname = "samples-{}.csv".format(fem_type)

        if n_elem == 10:
            df.to_csv(fname, mode="w", header="column_names", index=False)
        else:
            df.to_csv(fname, mode="a", header=False, index=False)

mesher_lin(1, 10)
