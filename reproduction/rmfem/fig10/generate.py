import numpy as np
import pandas as pd

from props.mcmc_props import mcmc_props
from props.rmfem_props import mwmc_props
from sampling import MCMCRunner
from rmfem.rmfemrunner import RMFEMRunner


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


obs_values = np.array(
    [
        0.01154101,
        0.01667733,
        0.01592942,
        0.00980423,
        -0.00043005,
        -0.01177105,
        -0.02001336,
        -0.0211289,
        -0.01350695,
    ]
)
n_obs = len(obs_values)
rng = np.random.default_rng(0)
corruption = rng.standard_normal(n_obs)

for fem_type in ["fem", "rmfem"]:
    if fem_type == "fem":
        props = mcmc_props
    elif fem_type == "rmfem":
        props = mwmc_props
    else:
        raise ValueError

    for n_elem in [10, 20, 40]:
        mesher_lin(1, n_elem)

        for std_noise in [1e-4, 1e-5, 1e-6]:
            if fem_type == "fem":
                likelihood_props = props["target"]["likelihood"]
            elif fem_type == "rmfem":
                likelihood_props = props["inner"]["target"]["likelihood"]
            else:
                raise ValueError

            likelihood_props["values"] = obs_values + std_noise * corruption
            likelihood_props["noise"]["cov"] = std_noise**2 * np.identity(n_obs)

            if fem_type == "fem":
                mcmc = MCMCRunner(**props)
                samples = mcmc()

                df = pd.DataFrame(samples, columns=["xi_1", "xi_2", "xi_3", "xi_4"])
                df["sample"] = df.index
                df["n_elem"] = n_elem
                df["std_noise"] = std_noise

            elif fem_type == "rmfem":
                rmfem = RMFEMRunner(**mwmc_props)
                samples = rmfem()

                subdf_list = []
                for i, sample in enumerate(samples):
                    subdf = pd.DataFrame(
                        sample, columns=["xi_1", "xi_2", "xi_3", "xi_4"]
                    )
                    subdf["sample"] = subdf.index
                    subdf["n_elem"] = n_elem
                    subdf["std_noise"] = std_noise
                    subdf["mesh"] = i
                    subdf_list.append(subdf)

                df = pd.concat(subdf_list)

            fname = "samples-{}.csv".format(fem_type)

            if n_elem == 10 and np.isclose(std_noise, 1e-4):
                df.to_csv(fname, mode="w", header="column_names", index=False)
            else:
                df.to_csv(fname, mode="a", header=False, index=False)

mesher_lin(1, 10)
