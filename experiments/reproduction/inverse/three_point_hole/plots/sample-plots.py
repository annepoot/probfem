import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D

from util.io import read_csv_from


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


refs_by_var = {
    "x": 1.0,
    "y": 0.4,
    "a": 0.4,
    "theta": np.pi / 6,
    "r_rel": 0.25,
    "r": 0.1,
}

N_filter = 100
N_start = 10000
N_end = 20000

std_corruption = 1e-4

for fem_type in ["fem", "bfem", "rmfem", "statfem"]:

    fname = "../samples-{}.csv".format(fem_type)

    df = read_csv_from(fname, "x,y,a,theta,r_rel")
    df = df[df["sample"] >= N_start]
    df = df[df["sample"] <= N_end]
    df = df[df["sample"] % N_filter == 0]
    df = df[abs(df["std_corruption"] - std_corruption) < 1e-8]
    df["theta"] = df["theta"] - (0.5 * np.pi) * np.floor(df["theta"] / (0.5 * np.pi))

    for h in [0.2, 0.1, 0.05]:
        # Create figure and axis
        fig, ax = plt.subplots()

        c = {0.2: "C0", 0.1: "C1", 0.05: "C2", 0.02: "C3"}[h]
        df_h = df[abs(df["h"] - h) < 1e-8]

        # Create a rounded rectangle
        rect = patches.FancyBboxPatch(
            (0, 0),  # Bottom-left corner
            5.0,
            1.0,
            boxstyle="Square, pad=0.0",
            edgecolor="black",
            facecolor="none",
            linewidth=1.0,
        )
        ax.add_patch(rect)

        for idx, row in df_h.iterrows():
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
        ax.set_xlim(0.0 - pad, 5.0 + pad)
        ax.set_ylim(0.0 - pad, 1.0 + pad)
        ax.set_aspect("equal")
        plt.axis("off")

        # Show the plot
        plt.show()
