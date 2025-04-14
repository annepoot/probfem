import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
import seaborn as sns

from util.io import read_csv_from


def get_hole_patch(x, y, a, theta, r):
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
    transform = Affine2D().rotate_deg_around(x, y, theta_deg) + plt.gca().transData
    patch.set_transform(transform)

    return patch


refs_by_var = {
    "x": 1.0,
    "y": 0.4,
    "a": 0.4,
    "theta": np.pi / 6,
    "r_rel": 0.25,
    "r": 0.1,
}

N_filter = 200
N_start = 10000
N_end = 20000

fem_types = ["fem", "bfem", "rmfem", "statfem"]
h_range = [0.2, 0.1, 0.05]
std_corruption = 1e-4

for fem_type in fem_types:

    fname = "output/samples-{}.csv".format(fem_type)

    df = read_csv_from(fname, "x,y,a,theta,r_rel")
    df = df[df["sample"] >= N_start]
    df = df[df["sample"] <= N_end]
    df = df[df["sample"] % N_filter == 0]
    df = df[abs(df["std_corruption"] - std_corruption) < 1e-8]
    df["theta"] = np.fmod(df["theta"], 0.5 * np.pi)

    for i, h in enumerate(h_range):
        # Create figure and axis
        fig, ax = plt.subplots()

        c = sns.color_palette("rocket_r", n_colors=8)[2 * i + 1]
        df_h = df[abs(df["h"] - h) < 1e-8]

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
            hole_patch = get_hole_patch(x, y, a, theta, r)
            hole_patch.set_edgecolor(c)
            hole_patch.set_alpha(0.5)
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
        pad = 0.01
        ax.set_xlim(0.0 - pad, 2.52)
        ax.set_ylim(-0.1 - pad, 1.15 + pad)
        ax.set_aspect("equal")
        ax.axvline(x=2.51, linestyle=(0, (3, 3, 1, 3)), color="black")

        plt.axis("off")

        fname = os.path.join("img", "sample-plot_{}_h-{:.2f}.pdf".format(fem_type, h))
        plt.savefig(fname, bbox_inches="tight")
        plt.show()
