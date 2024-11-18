import numpy as np
import pandas as pd

from mcmc_props import mcmc_props
from probability import MCMCRunner


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

for n_elem in [10, 20, 40]:
    mesher_lin(1, n_elem)

    for std_noise in [1e-4, 1e-5, 1e-6]:
        likelihood_props = mcmc_props["target"]["likelihood"]
        likelihood_props["values"] = obs_values + std_noise * corruption
        likelihood_props["noise"]["cov"] = std_noise**2 * np.identity(n_obs)

        mcmc = MCMCRunner(**mcmc_props)
        samples = mcmc()

        df = pd.DataFrame(samples, columns=["xi_1", "xi_2", "xi_3", "xi_4"])
        df["sample"] = df.index
        df["n_elem"] = n_elem
        df["std_noise"] = std_noise

        if n_elem == 10 and np.isclose(std_noise, 1e-4):
            df.to_csv("samples.csv", mode="w", header="column_names", index=False)
        else:
            df.to_csv("samples.csv", mode="a", header=False, index=False)

mesher_lin(1, 10)
