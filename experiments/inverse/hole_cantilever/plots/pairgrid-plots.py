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

N_filter = 100
N_start = 10000
N_end = 20000

for fem_type in ["fem", "rmfem"]:
    fname = "../samples-{}.csv".format(fem_type)
    df = read_csv_from(fname, "x,y,a,theta,r_rel")
    df = df[df["sample"] >= N_start]
    df = df[df["sample"] <= N_end]
    df = df[df["sample"] % N_filter == 0]
    df["theta"] = df["theta"] - (0.5 * np.pi) * np.floor(df["theta"] / (0.5 * np.pi))
    df["h"] = df["h"].astype(str)

    grid = sns.PairGrid(data=df, vars=variables, hue="h", diag_sharey=False, height=1.5)
    grid.map_upper(sns.scatterplot, alpha=0.5, marker=".", edgecolor=None)
    grid.map_lower(sns.kdeplot)
    grid.map_diag(sns.kdeplot)

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
