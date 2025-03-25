import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from util.io import read_csv_from


variables = ["rho", "log_l_d", "log_sigma_d"]

refs_by_var = {
    "rho": 1.0,
}

hard_lims_by_var = {
    "rho": (0.9, 1.1),
    "l_d": (1e-8, 1),
    "log_l_d": (np.log(1e-4), np.log(1e4)),
    "sigma_d": (1e-8, 1),
    "log_sigma_d": (np.log(1e-8), 0),
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
    "log_l_d": r"$l_d$",
    "log_sigma_d": r"$\sigma_d$",
}

width = 0.10
N_burn = 10000
N_filter = 50
h_range = [0.2, 0.1, 0.05]

fname = os.path.join("output", "samples-statfem.csv")
df = read_csv_from(fname, "x,y,a,theta,r_rel")
df = df[(df["sample"] >= N_burn) & (df["sample"] % N_filter == 0)]
df = df[df["h"].isin(h_range)]
df["h"] = df["h"].astype(str)
df["theta"] = np.fmod(df["theta"], np.pi / 2)

df = df.melt(id_vars=["h"], value_vars=variables)
df["fem_type"] = "statfem"

plt.rc("text", usetex=True)  # use latex for text
plt.rcParams["text.latex.preamble"] = r"\usepackage{xfrac}"

g = sns.FacetGrid(
    df,
    row="fem_type",
    col="variable",
    hue="h",
    height=2,
    margin_titles=False,
    sharex=False,
    sharey=False,
    palette=sns.color_palette("rocket_r", n_colors=8)[::2],
)

g.map_dataframe(sns.kdeplot, x="value", fill=False)

g.set_titles("")
g.add_legend(title=r"$h$")

for i, var in enumerate(variables):
    lims = lims_by_var(width)[var]
    xref = refs_by_var.get(var)

    ax = g.axes[0, i]

    if "log" in var:
        labels = [
            r"$10^{{{:d}}}$".format(int(lim))
            for lim in np.linspace(lims[0], lims[1], 3) / np.log(10)
        ]
    else:
        labels = None

    ax.set_xlim(lims)
    ax.set_xticks(np.linspace(lims[0], lims[1], 3), labels=labels)

    if xref is not None:
        ax.axvline(x=xref, color="k", label="ref", zorder=2)

    ax.set_xlabel(labels_by_var[var])
    ax.set_ylabel(None)

fname = os.path.join("img", "statfem-marginals.pdf")
plt.savefig(fname=fname, bbox_inches="tight")
plt.show()
