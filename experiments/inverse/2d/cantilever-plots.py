import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D


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


def get_hole_patch(x, y, a, theta, r):
    theta_deg = theta / np.pi * 180
    patch = patches.FancyBboxPatch(
        (x - a / 2, y - a / 2),
        a,
        a,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor="none",
        transform=Affine2D().rotate_deg_around(x, y, theta_deg) + ax.transData,
    )
    return patch


variables = ["x", "y", "a", "theta", "r_rel"]
refs_by_var = {
    "x": 1.0,
    "y": 0.4,
    "a": 0.4,
    "theta": np.pi / 6,
    "r_rel": 0.25,
    "r": 0.1,
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

N_filter = 1000
N_start = 10000
N_end = 50000


for fem_type in ["fem"]:
    for width in [0.1]:

        # Create figure and axis
        fig, ax = plt.subplots()

        for h, c in zip([0.2, 0.1, 0.05], ["C0", "C1", "C2"]):
            fname = "samples-{}.csv".format(fem_type)
            df = read_csv_from(fname, "x,y,a,theta,r_rel")
            df = df[df["sample"] >= N_start]
            df = df[df["sample"] <= N_end]
            df = df[df["sample"] % N_filter == 0]
            df = df[abs(df["h"] - h) < 1e-8]

            df["theta"] = df["theta"] - (0.5 * np.pi) * np.floor(
                df["theta"] / (0.5 * np.pi)
            )

            row = df.iloc[0]

            # Create a rounded rectangle
            rect = patches.FancyBboxPatch(
                (0, 0),  # Bottom-left corner
                4.0,
                1.0,
                boxstyle="Square, pad=0.0",
                edgecolor="black",
                facecolor="none",
                linewidth=1.0,
            )
            ax.add_patch(rect)

            for idx, row in df.iterrows():
                x, y, a, theta, r = row[["x", "y", "a", "theta", "r"]]
                hole_patch = get_hole_patch(x, y, a, theta, r)
                hole_patch.set_edgecolor(c)
                hole_patch.set_alpha(0.1)
                hole_patch.set_linewidth(0.5)
                ax.add_patch(hole_patch)

        x_ref = refs_by_var["x"]
        y_ref = refs_by_var["y"]
        a_ref = refs_by_var["a"]
        theta_ref = refs_by_var["theta"]
        r_ref = refs_by_var["r"]

        ref_patch = get_hole_patch(x_ref, y_ref, a_ref, theta_ref, r_ref)
        ref_patch.set_edgecolor("black")
        ref_patch.set_alpha(1.0)
        ref_patch.set_linewidth(1.0)
        ax.add_patch(ref_patch)

        # Set limits and aspect ratio
        pad = 0.1
        ax.set_xlim(0.0 - pad, 4.0 + pad)
        ax.set_ylim(0.0 - pad, 1.0 + pad)
        ax.set_aspect("equal")
        plt.axis("off")

        # Show the plot
        plt.show()
