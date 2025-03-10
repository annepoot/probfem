import os
import numpy as np
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

# plims_by_fem_type = {
#     "fem": 150,
#     "rmfem": 25,
#     "bfem": 25,
#     "statfem": 25,
# }


def lims_by_var(width):
    lims_by_var = {}
    for var in variables:
        ref = refs_by_var[var]
        lims_by_var[var] = (ref - width, ref + width)
    return lims_by_var


labels_by_var = {
    "x": r"$x$",
    "y": r"$y$",
    "a": r"$a$",
    "theta": r"$\theta$",
    "r_rel": r"$r$",
}

width = 0.10
N_burn = 10000
N_filter = 50
h_range = [0.2, 0.1, 0.05]


def custom_kde_2d(x, y, *, color, label, **kwargs):
    sns.kdeplot(x=x, y=y, color=color, label=label, **kwargs)


plt.rc("text", usetex=True)  # use latex for text
plt.rcParams["text.latex.preamble"] = r"\usepackage{xfrac}"

for fem_type in ["fem", "bfem", "rmfem"]:
    fname = os.path.join("..", "samples-{}.csv".format(fem_type))
    df = read_csv_from(fname, "x,y,a,theta,r_rel")
    df = df[(df["sample"] >= N_burn) & (df["sample"] % N_filter == 0)]
    df = df[df["h"].isin(h_range)]
    df["h"] = df["h"].astype(str)
    df["theta"] = np.fmod(df["theta"], np.pi / 2)

    grid = sns.PairGrid(data=df, vars=variables, hue="h", diag_sharey=True, height=1.5)
    grid.map_upper(sns.scatterplot, alpha=0.5, marker=".", linewidths=0.0)
    grid.map_lower(custom_kde_2d, levels=5)
    grid.map_diag(sns.kdeplot)

    # nvar = len(variables)
    # for i, var in enumerate(variables):
    #     lims = lims_by_var(width)[var]
    #     grid.axes[(i + 1) % nvar, i].set_xlim(lims)
    #     grid.axes[i, (i + 1) % nvar].set_ylim(lims)
    #     grid.axes[(i + 1) % nvar, i].set_xticks(np.linspace(lims[0], lims[1], 3))
    #     grid.axes[i, (i + 1) % nvar].set_yticks(np.linspace(lims[0], lims[1], 3))
    #
    # for i in range(len(variables)):
    #     grid.diag_axes[i].set_ylim((0, plims_by_fem_type[fem_type]))

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

    # labels = [r"$\sfrac{1}{10}$", r"$\sfrac{1}{20}$", r"$\sfrac{1}{40}$"]

    grid.add_legend(title=r"$h$")
    fontsize = "12"
    plt.setp(grid.legend.get_texts(), fontsize=fontsize)
    plt.setp(grid.legend.get_title(), fontsize=fontsize)
