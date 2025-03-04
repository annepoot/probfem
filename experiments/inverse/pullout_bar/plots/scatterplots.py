import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from util.io import read_csv_from


variables = ["E", "k"]
refs_by_var = {
    "E": 1.0,
    "k": 100.0,
}

plims_by_fem_type = {
    "fem": 150,
    "rmfem": 25,
    "bfem": 25,
    "statfem": 25,
}


def lims_by_var(width):
    lims_by_var = {}
    for var in variables:
        ref = refs_by_var[var]
        lims_by_var[var] = (ref * (1 - width), ref * (1 + width))
    return lims_by_var


labels_by_var = {
    "E": r"$E$",
    "k": r"$k$",
}

width = 1.0
N_burn = 10000
N_filter = 5
N_elem_range = [1, 2, 4, 8, 16, 32, 64]


plt.rc("text", usetex=True)  # use latex for text
plt.rcParams["text.latex.preamble"] = r"\usepackage{xfrac}"

for fem_type in ["fem", "rmfem"]:
    fname = os.path.join("..", "samples-{}.csv".format(fem_type))
    df = read_csv_from(fname, "E,k")
    df = df[(df["sample"] >= N_burn) & (df["sample"] % N_filter == 0)]
    df = df[df["n_elem"].isin(N_elem_range)]
    df["n_elem"] = df["n_elem"].astype(str)
    df["h"] = r"\sfrac{1}{" + df["n_elem"] + "}"

    fig, ax = plt.subplots(figsize=(4, 4))
    plot = sns.scatterplot(
        data=df, x="E", y="k", hue="h", alpha=0.5, marker=".", linewidths=0.0, ax=ax
    )

    ref_E = refs_by_var["E"]
    ref_k = refs_by_var["k"]
    lims_E = lims_by_var(width)["E"]
    lims_k = lims_by_var(width)["k"]

    ax.set_xlim(lims_E)
    ax.set_ylim(lims_k)
    ax.set_xticks(np.linspace(lims_E[0], lims_E[1], 5))
    ax.set_yticks(np.linspace(lims_k[0], lims_k[1], 5))
    ax.xaxis.set_label_text(labels_by_var["E"])
    ax.yaxis.set_label_text(labels_by_var["k"])

    ax.scatter([ref_E], [ref_k], color="k", zorder=2)

    for n_elem in N_elem_range:
        subdf = df[df["n_elem"] == str(n_elem)]
        nu = np.sqrt(subdf["k"] / subdf["E"])
        mean = np.mean(nu)
        std = np.std(nu)

    ax.legend(title=r"$h$")
    plt.show()
