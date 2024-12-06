import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

variables = {
    "fem": ["sigma_e"],
    "bfem": ["scale"],
    "statfem": ["rho", "l_d", "sigma_d"],
}

labels_by_var = {
    "rho": r"$\rho$",
    "l_d": r"$\l_d$",
    "sigma_d": r"$\sigma_d$",
    "sigma_e": r"$\sigma_e$",
}

N_burn = 5000
N_filter = 1

for fem_type in ["fem", "bfem", "statfem"]:
    fname = "samples-{}-hyper.csv".format(fem_type)
    df = pd.read_csv(fname)
    df["n_elem"] = df["n_elem"].astype(str)

    for var in variables[fem_type]:
        plt.figure()
        ax = sns.kdeplot(
            df[(df["sample"] >= N_burn)], x=var, hue="n_elem", log_scale=True
        )
        ax.set_xlabel(labels_by_var[var])
        plt.show()

    df = df[df["n_elem"] == "40"]
    df = df[df["sample"] > N_burn]

    for i, var1 in enumerate(variables[fem_type]):
        for var2 in variables[fem_type][i + 1 :]:
            plt.figure()
            ax = sns.scatterplot(
                df, x=var1, y=var2, hue="n_elem", alpha=0.5, marker=".", edgecolor=None
            )
            ax.set_xlabel(labels_by_var[var1])
            ax.set_ylabel(labels_by_var[var2])
            plt.show()
