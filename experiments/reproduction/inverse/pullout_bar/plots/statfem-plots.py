import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from util.io import read_csv_from

variables = ["rho", "l_d", "sigma_d"]
refs_by_var = {
    "rho": 1.0,
}

hard_lims_by_var = {
    "l_d": (1e-6, 1e0),
    "sigma_d": (1e-8, 1e-2),
}

scales_by_var = {
    "rho": "linear",
    "l_d": "log",
    "sigma_d": "log",
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
    "rho": r"$\rho$",
    "l_d": r"$l_d$",
    "sigma_d": r"$\sigma_d$",
}

width = 0.10
N_burn = 10000
N_filter = 50
N_elem_range = [64]


def custom_kde_2d(x, y, *, color, label, **kwargs):
    sns.kdeplot(x=x, y=y, color=color, label=label, **kwargs)


plt.rc("text", usetex=True)  # use latex for text
plt.rcParams["text.latex.preamble"] = r"\usepackage{xfrac}"

for fem_type in ["statfem"]:
    fname = os.path.join("..", "samples-{}.csv".format(fem_type))
    df = read_csv_from(fname, "E,k")
    df = df[(df["sample"] >= N_burn) & (df["sample"] % N_filter == 0)]
    df = df[df["n_elem"].isin(N_elem_range)]
    df["n_elem"] = df["n_elem"].astype(str)
    df["h"] = r"\sfrac{1}{" + df["n_elem"] + "}"

    grid = sns.PairGrid(data=df, vars=variables, hue="h", diag_sharey=False, height=1.5)
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

    for i, xvar in enumerate(variables):
        for j, yvar in enumerate(variables):
            xlabel = labels_by_var[xvar]
            ylabel = labels_by_var[yvar]
            grid.axes[j, i].xaxis.set_label_text(xlabel)
            grid.axes[j, i].yaxis.set_label_text(ylabel)

            xref = refs_by_var.get(xvar)
            yref = refs_by_var.get(yvar)

            if i == j:
                if xref is not None:
                    grid.axes[j, i].axvline(x=xref, color="k", label="ref", zorder=2)
            else:
                if xref is not None:
                    if yref is not None:
                        grid.axes[j, i].scatter(
                            [xref], [yref], color="k", label="ref", zorder=2
                        )
                    else:
                        grid.axes[j, i].axvline(
                            x=xref, color="k", label="ref", zorder=2
                        )
                else:
                    if yref is not None:
                        grid.axes[j, i].axhline(
                            y=yref, color="k", label="ref", zorder=2
                        )

            xscale = scales_by_var[xvar]
            yscale = scales_by_var[yvar]

            grid.axes[j, i].set_xscale(xscale)
            grid.axes[j, i].set_yscale(yscale)

    # labels = [r"$\sfrac{1}{10}$", r"$\sfrac{1}{20}$", r"$\sfrac{1}{40}$"]

    grid.add_legend(title=r"$h$")
    fontsize = "12"
    plt.setp(grid.legend.get_texts(), fontsize=fontsize)
    plt.setp(grid.legend.get_title(), fontsize=fontsize)
