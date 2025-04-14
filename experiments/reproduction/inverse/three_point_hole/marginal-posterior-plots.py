import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from util.io import read_csv_from


variables = ["x", "y", "a", "theta", "r_rel"]

refs_by_var = {
    "x": 1.0,
    "y": 0.4,
    "a": 0.4,
    "theta": np.pi / 6,
    "r_rel": 0.25,
}

hard_lims_by_var = {
    "theta": (0.0, np.pi / 2),
    "r_rel": (0.0, 0.5),
}

p_lims_by_var = {
    "x": 20,
    "y": 20,
    "a": 50,
    "theta": 8,
    "r_rel": 25,
}


def lims_by_var(width):
    lims_by_var = {}
    for var in variables:
        if var in hard_lims_by_var:
            lims_by_var[var] = hard_lims_by_var[var]
        else:
            ref = refs_by_var[var]
            lims_by_var[var] = (ref - width, ref + width)
    return lims_by_var


labels_by_var = {
    "x": r"$x$",
    "y": r"$y$",
    "a": r"$d$",
    "theta": r"$\alpha$",
    "r_rel": r"$r$",
    "fem": r"FEM",
    "bfem": r"BFEM",
    "rmfem": r"RM-FEM",
    "statfem": r"statFEM",
}

width = 0.10
N_burn = 10000
N_filter = 50

fem_types = ["fem", "bfem", "rmfem", "statfem"]
h_range = [0.2, 0.1, 0.05]
std_corruption = 1e-4

dfs = []
for fem_type in fem_types:
    fname = os.path.join("output", "samples-{}.csv".format(fem_type))
    df = read_csv_from(fname, "x,y,a,theta,r_rel")
    df = df[(df["sample"] >= N_burn) & (df["sample"] % N_filter == 0)]
    df = df[abs(df["std_corruption"] - std_corruption) < 1e-8]
    df = df[df["h"].isin(h_range)]
    df["h"] = df["h"].astype(str)
    df["theta"] = np.fmod(df["theta"], np.pi / 2)
    df = df.melt(id_vars=["h"], value_vars=["x", "y", "a", "theta", "r_rel"])
    df["fem_type"] = fem_type
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

plt.rc("text", usetex=True)  # use latex for text
plt.rcParams["text.latex.preamble"] = r"\usepackage{xfrac}"

g = sns.FacetGrid(
    df_all,
    row="fem_type",
    col="variable",
    hue="h",
    height=2,
    margin_titles=False,
    sharex=False,
    sharey=False,
    palette=sns.color_palette("rocket_r", n_colors=8)[1::2],
)

g.map_dataframe(sns.kdeplot, x="value", fill=False)

g.set_titles("")
g.add_legend(title=r"$h$")

for i, var in enumerate(variables):
    lims = lims_by_var(width)[var]
    plims = (0, p_lims_by_var[var])
    xref = refs_by_var[var]

    for j, fem_type in enumerate(fem_types):
        ax = g.axes[j, i]

        if var == "theta":
            labels = [r"$0$", r"$\sfrac{\pi}{4}$", r"$\sfrac{\pi}{2}$"]
        else:
            labels = None

        ax.set_xlim(lims)
        ax.set_xticks(np.linspace(lims[0], lims[1], 3), labels=labels)
        ax.set_ylim(plims)
        ax.set_yticks(np.linspace(plims[0], plims[1], 3))

        if xref is not None:
            ax.axvline(x=xref, color="k", label="ref", zorder=2)

        if j == len(fem_types) - 1:
            ax.set_xlabel(labels_by_var[var])
        else:
            ax.set_xlabel(None)

        if i == 0:
            ax.set_ylabel(labels_by_var[fem_type])
        else:
            ax.set_ylabel(None)

fname = os.path.join("img", "posterior-marginals.pdf")
plt.savefig(fname=fname, bbox_inches="tight")
plt.show()
