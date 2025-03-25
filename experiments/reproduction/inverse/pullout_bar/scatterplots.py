import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from util.io import read_csv_from


variables = ["E", "k"]
refs_by_var = {
    "E": 0.8,
    "k": 70.0,
}


def lims_by_var(width):
    lims_by_var = {}
    for var in variables:
        ref = refs_by_var[var]
        lims_by_var[var] = (ref * (1 - width), ref * (1 + width))
    return lims_by_var


labels_by_var = {
    "E": r"$EA$",
    "k": r"$k$",
}

width = 1.0
N_burn = 10000
N_filter = 50

plt.rc("text", usetex=True)  # use latex for text
plt.rcParams["text.latex.preamble"] = r"\usepackage{xfrac}"
colors = dict(zip([1, 2, 4, 8, 16, 32, 64], sns.color_palette("rocket_r", n_colors=8)))

for fem_type in ["fem", "bfem", "rmfem", "statfem"]:
    if fem_type == "fem":
        n_elem_range = [1, 2, 4, 8, 16, 32, 64]
    else:
        n_elem_range = [1, 4, 16, 64]

    fname = os.path.join("output", "samples-{}.csv".format(fem_type))
    df = read_csv_from(fname, "log_E,log_k")
    df = df[(df["sample"] >= N_burn) & (df["sample"] % N_filter == 0)]
    df = df[df["n_elem"].isin(n_elem_range)]
    df["n_elem"] = df["n_elem"].astype(str)
    df["h"] = r"\sfrac{1}{" + df["n_elem"] + "}"

    fig, ax = plt.subplots(figsize=(4, 4))

    plot = sns.scatterplot(
        data=df,
        x="E",
        y="k",
        hue="h",
        alpha=0.6,
        marker=".",
        linewidths=0.0,
        ax=ax,
        palette=[colors[n_elem] for n_elem in n_elem_range],
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
    E = np.linspace(lims_E[0] + 1e-8, lims_E[1], 1000)
    k = ref_E * ref_k / E
    ax.plot(E, k, color="k", linestyle=":")

    legend = ax.legend(title=r"$h$")
    fontsize = "12"
    plt.setp(legend.get_texts(), fontsize=fontsize)
    plt.setp(legend.get_title(), fontsize=fontsize)
    for handle in legend.legendHandles:
        handle.set_alpha(1.0)

    fname = os.path.join("img", "scatterplot_{}.pdf".format(fem_type))
    plt.savefig(fname=fname, bbox_inches="tight")
    plt.show()
