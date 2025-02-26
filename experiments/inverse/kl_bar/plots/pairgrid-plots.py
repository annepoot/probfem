import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def read_csv_from(fname, line, **kwargs):
    with open(fname) as f:
        pos = 0
        cur_line = f.readline()
        while not cur_line.startswith(line):
            pos = f.tell()
            cur_line = f.readline()
        f.seek(pos)
        return pd.read_csv(f, **kwargs)


variables = ["xi_1", "xi_2", "xi_3", "xi_4"]
refs_by_var = {
    "xi_1": 1.0,
    "xi_2": 1.0,
    "xi_3": 0.25,
    "xi_4": 0.25,
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
        lims_by_var[var] = (ref - width, ref + width)
    return lims_by_var


labels_by_var = {
    "xi_1": r"$\xi_1$",
    "xi_2": r"$\xi_2$",
    "xi_3": r"$\xi_3$",
    "xi_4": r"$\xi_4$",
}

width = 0.10
N_burn = 10000
N_filter = 20
N_elem_range = [10, 20, 40]


def custom_kde_2d(x, y, *, color, label, **kwargs):
    sns.kdeplot(x=x, y=y, color=color, label=label, **kwargs)


plt.rc("text", usetex=True)  # use latex for text
plt.rcParams["text.latex.preamble"] = r"\usepackage{xfrac}"

for fem_type in ["fem", "bfem", "rmfem", "statfem"]:
    fname = os.path.join("..", "samples-{}.csv".format(fem_type))
    df = read_csv_from(fname, "xi_1,xi_2,xi_3,xi_4")
    df = df[(df["sample"] >= N_burn) & (df["sample"] % N_filter == 0)]
    df = df[df["n_elem"].isin(N_elem_range)]
    df["n_elem"] = df["n_elem"].astype(str)
    df["h"] = r"\sfrac{1}{" + df["n_elem"] + "}"

    grid = sns.PairGrid(data=df, vars=variables, hue="h", diag_sharey=True, height=1.5)
    grid.map_upper(sns.scatterplot, alpha=0.5, marker=".", linewidths=0.0)
    grid.map_lower(custom_kde_2d, levels=5)
    grid.map_diag(sns.kdeplot)

    nvar = len(variables)
    for i, var in enumerate(variables):
        lims = lims_by_var(width)[var]
        grid.axes[(i + 1) % nvar, i].set_xlim(lims)
        grid.axes[i, (i + 1) % nvar].set_ylim(lims)
        grid.axes[(i + 1) % nvar, i].set_xticks(np.linspace(lims[0], lims[1], 3))
        grid.axes[i, (i + 1) % nvar].set_yticks(np.linspace(lims[0], lims[1], 3))

    for i in range(len(variables)):
        grid.diag_axes[i].set_ylim((0, plims_by_fem_type[fem_type]))

    for i, xvar in enumerate(variables):
        for j, yvar in enumerate(variables):
            xlabel = labels_by_var[xvar]
            ylabel = labels_by_var[yvar]
            grid.axes[j, i].xaxis.set_label_text(xlabel)
            grid.axes[j, i].yaxis.set_label_text(ylabel)

            xref = refs_by_var[xvar]
            yref = refs_by_var[yvar]
            if i == j:
                grid.axes[j, i].axvline(x=xref, color="k", label="ref", zorder=2)
            else:
                grid.axes[j, i].scatter(
                    [xref], [yref], color="k", label="ref", zorder=2
                )

    labels = [r"$\sfrac{1}{10}$", r"$\sfrac{1}{20}$", r"$\sfrac{1}{40}$"]

    grid.add_legend(title=r"$h$")
    fontsize = "12"
    plt.setp(grid.legend.get_texts(), fontsize=fontsize)
    plt.setp(grid.legend.get_title(), fontsize=fontsize)
