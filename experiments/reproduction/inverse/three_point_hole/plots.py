import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
import seaborn as sns
import pandas as pd

from probability.multivariate import Gaussian
from util.io import read_csv_from, unique_filename

plt.rc("text", usetex=True)  # use latex for text
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern Roman"]
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["text.latex.preamble"] = r"\usepackage{xfrac}"

refs_by_var = {
    "x": 1.0,
    "y": 0.4,
    "a": 0.4,
    "theta": np.pi / 6,
    "r_rel": 0.25,
    "r": 0.1,
    "rho": 1.0,
}


hard_lims_by_var = {
    "theta": (0.0, np.pi / 2),
    "r_rel": (0.0, 0.5),
    "rho": (0.9, 1.1),
    "l_d": (1e-8, 1),
    "log_l_d": (np.log(1e-4), np.log(1e4)),
    "sigma_d": (1e-8, 1),
    "log_sigma_d": (np.log(1e-8), 0),
}


def lims_by_var(width):
    lims_by_var = {}

    for var, ref in refs_by_var.items():
        lims_by_var[var] = (ref - width, ref + width)

    for var, bounds in hard_lims_by_var.items():
        lims_by_var[var] = bounds

    return lims_by_var

p_lims_by_var = {
    "x": 20,
    "y": 20,
    "a": 30,
    "theta": 4,
    "r_rel": 10,
}


labels_by_var = {
    "x": r"$x$",
    "y": r"$y$",
    "a": r"$d$",
    "theta": r"$\alpha$",
    "r_rel": r"$r$",
    "rho": r"$\rho$",
    "l_d": r"$l_d$",
    "sigma_d": r"$\sigma_d$",
    "log_l_d": r"$l_d$",
    "log_sigma_d": r"$\sigma_d$",
    "fem": r"FEM",
    "bfem": r"BFEM",
    "rmfem": r"RM-FEM",
    "statfem": r"statFEM",
}


def get_hole_patch(x, y, a, theta, r, ax):
    theta_deg = theta / np.pi * 180
    k = 0.55228 * r  # Approximate control point distance for circular Bézier

    # Path data (moving counterclockwise)
    path_data = [
        (Path.MOVETO, (x + a / 2 - r, y - a / 2)),  # Start bottom-right corner
        # Bottom-right corner arc (270° to 360°)
        (Path.CURVE4, (x + a / 2 - r + k, y - a / 2)),
        (Path.CURVE4, (x + a / 2, y - a / 2 + r - k)),
        (Path.CURVE4, (x + a / 2, y - a / 2 + r)),
        (Path.LINETO, (x + a / 2, y + a / 2 - r)),  # Right side
        # Top-right corner arc (0° to 90°)
        (Path.CURVE4, (x + a / 2, y + a / 2 - r + k)),
        (Path.CURVE4, (x + a / 2 - r + k, y + a / 2)),
        (Path.CURVE4, (x + a / 2 - r, y + a / 2)),
        (Path.LINETO, (x - a / 2 + r, y + a / 2)),  # Top side
        # Top-left corner arc (90° to 180°)
        (Path.CURVE4, (x - a / 2 + r - k, y + a / 2)),
        (Path.CURVE4, (x - a / 2, y + a / 2 - r + k)),
        (Path.CURVE4, (x - a / 2, y + a / 2 - r)),
        (Path.LINETO, (x - a / 2, y - a / 2 + r)),  # Left side
        # Bottom-left corner arc (180° to 270°)
        (Path.CURVE4, (x - a / 2, y - a / 2 + r - k)),
        (Path.CURVE4, (x - a / 2 + r - k, y - a / 2)),
        (Path.CURVE4, (x - a / 2 + r, y - a / 2)),
        (Path.CLOSEPOLY, (x + a / 2 - r, y - a / 2)),  # Close the path
    ]

    codes, verts = zip(*path_data)
    path = Path(verts, codes)

    # Create the PathPatch
    patch = PathPatch(path, edgecolor="blue", facecolor="none", lw=2)

    # Apply rotation
    transform = Affine2D().rotate_deg_around(x, y, theta_deg) + ax.transData
    patch.set_transform(transform)

    return patch


def sample_plot(df, save=False):
    colors = dict(zip([0.20, 0.15, 0.10, 0.08, 0.05, 0.02, 0.01], sns.color_palette("rocket_r", n_colors=8)[1:]))
    h_range = np.array(df["h"].drop_duplicates())

    # Create figure and axis
    n_plot = len(h_range)

    if not save:
        fig, axs = plt.subplots(ncols=n_plot, figsize=(6.4 * n_plot, 4.8))

    for i, h in enumerate(h_range):
        if save:
            fig, ax = plt.subplots()
        else:
            ax = axs[i]

        c = colors[h]
        df_h = df[df["h"] == h]

        # Create a rounded rectanglepatch
        rect = FancyBboxPatch(
            (0, 0),  # Bottom-left corner
            5.0,
            1.0,
            boxstyle="Square, pad=0.0",
            edgecolor="black",
            facecolor="none",
            linewidth=1.0,
        )
        ax.add_patch(rect)

        # Create a rounded rectanglepatch
        for x, y in zip([0.5, 2.5, 4.5], [0.0, 1.1, 0.0]):
            support = FancyBboxPatch(
                (x - 0.1, y - 0.1),  # Bottom-left corner
                0.2,
                0.1,
                boxstyle="Square, pad=0.0",
                edgecolor="black",
                facecolor="0.7",
                linewidth=1.0,
            )
            ax.add_patch(support)

        for idx, row in df_h.iterrows():
            x, y, a, theta, r = row[["x", "y", "a", "theta", "r"]]
            hole_patch = get_hole_patch(x, y, a, theta, r, ax)
            hole_patch.set_edgecolor(c)
            hole_patch.set_alpha(0.5)
            hole_patch.set_linewidth(0.5)
            ax.add_patch(hole_patch)

        x_ref = refs_by_var["x"]
        y_ref = refs_by_var["y"]
        a_ref = refs_by_var["a"]
        theta_ref = refs_by_var["theta"]
        r_ref = refs_by_var["r"]

        ref_patch = get_hole_patch(x_ref, y_ref, a_ref, theta_ref, r_ref, ax)
        ref_patch.set_edgecolor("black")
        ref_patch.set_alpha(1.0)
        ref_patch.set_linewidth(1.0)
        ax.add_patch(ref_patch)

        # Set limits and aspect ratio
        pad = 0.01
        ax.set_xlim(0.0 - pad, 2.52)
        ax.set_ylim(-0.1 - pad, 1.15 + pad)
        ax.set_aspect("equal")
        ax.axvline(x=2.51, linestyle=(0, (3, 3, 1, 3)), color="black")
        ax.axis("off")

        if save:
            fname = unique_filename("sample-plot.pdf")
            plt.savefig(fname=fname, bbox_inches="tight")
            plt.show()

    if not save:
        plt.show()


def marginal_plot(dfs, save=False):
    width = 0.20

    df_list = []
    for fem_type, df in dfs.items():
        df["h"] = df["h"].astype(str)

        if "theta" in df:
            df["theta"] = np.fmod(df["theta"], np.pi / 2)

        df = df.melt(id_vars=["h"], value_vars=df.columns.drop("h"))
        df["fem_type"] = fem_type
        df_list.append(df)

    df_all = pd.concat(df_list, ignore_index=True)
    colors = dict(zip([0.20, 0.15, 0.10, 0.08, 0.05, 0.02, 0.01], sns.color_palette("rocket_r", n_colors=8)[1:]))
    h_range = np.array(df["h"].drop_duplicates())
    palette = [colors[float(h)] for h in h_range]

    g = sns.FacetGrid(
        df_all,
        row="fem_type",
        col="variable",
        hue="h",
        height=2,
        margin_titles=False,
        sharex=False,
        sharey=False,
        palette=palette,
    )

    g.map_dataframe(sns.kdeplot, x="value", fill=False)
    g.set_titles("")

    for i, var in enumerate(df_all["variable"].drop_duplicates()):
        for j, fem_type in enumerate(dfs.keys()):
            ax = g.axes[j, i]
            lims = lims_by_var(width)[var]

            if var == "theta":
                labels = [r"$0$", r"$\sfrac{\pi}{4}$", r"$\sfrac{\pi}{2}$"]
            elif "log" in var:
                labels = [
                    r"$10^{{{:d}}}$".format(int(lim))
                    for lim in np.linspace(lims[0], lims[1], 3) / np.log(10)
                ]
            else:
                labels = None

            ax.set_xlim(lims)
            ax.set_xticks(np.linspace(lims[0], lims[1], 3), labels=labels)

            if var in p_lims_by_var:
                plims = (0, p_lims_by_var[var])
                ax.set_ylim(plims)
                ax.set_yticks(np.linspace(plims[0], plims[1], 3))

            if var in refs_by_var:
                xref = refs_by_var[var]
                ax.axvline(x=xref, color="k", label="truth", zorder=2)

            if j == len(dfs) - 1:
                ax.set_xlabel(labels_by_var[var])
            else:
                ax.set_xlabel(None)

            if i == 0:
                ax.set_ylabel(labels_by_var[fem_type])
            else:
                ax.set_ylabel(None)

        if var == "rho":
            x = np.linspace(lims[0], lims[1], 100)
            logx = np.log(x)
            pdf = Gaussian(np.log(1), 0.1).calc_pdf(logx)
            ax.plot(x, pdf, color="0.7", zorder=0.5)
        elif var == "log_l_d":
            logx = np.linspace(lims[0], lims[1], 100)
            pdf = Gaussian(np.log(1), np.log(1e1)).calc_pdf(logx)
            ax.plot(logx, pdf, color="0.7", zorder=0.5)
        elif var == "log_sigma_d":
            logx = np.linspace(lims[0], lims[1], 100)
            pdf = Gaussian(np.log(1e-4), np.log(1e1)).calc_pdf(logx)
            ax.plot(logx, pdf, color="0.7", zorder=0.5, label="prior")

    ax = g.axes[g.axes.shape[0] // 2, g.axes.shape[1] - 1]

    ax.legend(title=r"$h$", bbox_to_anchor=(1.0, 1.0), loc="center left", frameon=False)

    if save:
        fname = unique_filename("marginal-plot.pdf")
        plt.savefig(fname=fname, bbox_inches="tight")

    plt.show()
