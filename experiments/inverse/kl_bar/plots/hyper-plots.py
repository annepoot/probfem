import matplotlib.pyplot as plt
import seaborn as sns

from util.io import read_csv_from


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
    "scale": r"$\alpha$",
}

N_filter = 100
N_burn = 5000

for fem_type in ["fem", "bfem", "statfem"]:
    fname = "../samples-{}-hyperprior.csv".format(fem_type)
    df = read_csv_from(fname, "xi_1,xi_2,xi_3,xi_4")
    df = df[(df["sample"] >= N_burn) & (df["sample"] % N_filter == 0)]
    df["n_elem"] = df["n_elem"].astype(str)

    for var in variables[fem_type]:
        plt.figure()
        ax = sns.kdeplot(
            df[(df["sample"] >= N_burn)], x=var, hue="n_elem", log_scale=True
        )
        ax.set_xlabel(labels_by_var[var])
        plt.show()

    df = df[df["n_elem"] == "40"]

    for i, var1 in enumerate(variables[fem_type]):
        for var2 in variables[fem_type][i + 1 :]:
            plt.figure()
            ax = sns.scatterplot(
                df, x=var1, y=var2, hue="n_elem", alpha=0.5, marker=".", edgecolor=None
            )
            ax.set_xlabel(labels_by_var[var1])
            ax.set_ylabel(labels_by_var[var2])
            plt.show()
