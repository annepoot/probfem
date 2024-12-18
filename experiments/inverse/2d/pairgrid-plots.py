import numpy as np
import pandas as pd
import seaborn as sns


def read_csv_from(fname, line, **kwargs):
    with open(fname) as f:
        pos = 0
        cur_line = f.readline()
        while not cur_line.startswith(line):
            pos = f.tell()
            cur_line = f.readline()
            if cur_line == "":
                raise EOFError("Line not found!")
        f.seek(pos)
        return pd.read_csv(f, **kwargs)


variables = ["x", "y", "a", "theta", "r_rel"]
refs_by_var = {
    "x": 1.0,
    "y": 0.4,
    "a": 0.4,
    "theta": np.pi / 6,
    "r_rel": 0.25,
}


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
    "r_rel": r"$r_{rel}$",
}

title_map = {
    "fem": "FEM",
    "bfem": "BFEM",
    "statfem": "StatFEM",
    "rmfem": "RM-FEM",
}

N_filter = 10
N_burn = 5000

for fem_type in ["fem"]:
    for width in [0.1]:
        fname = "samples-{}.csv".format(fem_type)
        df = read_csv_from(fname, "x,y,a,theta,r_rel")
        df = df[(df["sample"] >= N_burn) & (df["sample"] % N_filter == 0)]
        df["theta"] = df["theta"] - (0.5 * np.pi) * np.floor(
            df["theta"] / (0.5 * np.pi)
        )
        df["h"] = df["h"].astype(str)

        grid = sns.PairGrid(
            data=df, vars=variables, hue="h", diag_sharey=False, height=1.5
        )
        grid.map_upper(sns.scatterplot, alpha=0.5, marker=".", edgecolor=None)
        grid.map_lower(sns.kdeplot)
        grid.map_diag(sns.kdeplot)

        nvar = len(variables)
        for i, var in enumerate(variables):
            lims = lims_by_var(width)[var]
            # grid.axes[(i + 1) % nvar, i].set_xlim(lims)
            # grid.axes[i, (i + 1) % nvar].set_ylim(lims)
            # grid.axes[(i + 1) % nvar, i].set_xticks(np.linspace(lims[0], lims[1], 3))
            # grid.axes[i, (i + 1) % nvar].set_yticks(np.linspace(lims[0], lims[1], 3))

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

        grid.add_legend()
        grid.fig.subplots_adjust(top=0.95)
        grid.fig.suptitle(title_map[fem_type])
        # grid.savefig(
        #     fname="img/pairgrid-plot_noise-{}_width-{}".format(noise, int(width * 100)),
        #     dpi=300,
        # )
