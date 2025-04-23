import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
import seaborn as sns

from util.io import read_csv_from

from experiments.reproduction.inverse.three_point_hole.meshing import (
    create_mesh,
    get_observation_locations,
)


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

# Create figure and axis
fig, ax = plt.subplots(figsize=(9.0, 2.4), tight_layout=True)

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

nodes, elems = create_mesh(
    h=0.2,
    L=5.0,
    H=1.0,
    U=0.5,
    x=refs_by_var["x"],
    y=refs_by_var["y"],
    a=refs_by_var["a"],
    theta=refs_by_var["theta"],
    r_rel=refs_by_var["r_rel"],
    h_meas=0.5,
)

edges = []
for inodes in elems:
    for i in range(len(inodes)):
        edges.append(inodes[[i, (i + 1) % 3]])
edges = np.unique(np.sort(np.array(edges), axis=1), axis=0)

for edge in edges:
    coords = nodes[edge]
    ax.plot(coords[:, 0], coords[:, 1], color="0.7", linewidth=0.5, zorder=0)

obs_locs = get_observation_locations(L=5.0, H=1.0, h_meas=0.5)
ax.scatter(obs_locs[:, 0], obs_locs[:, 1], marker=".", color="k", zorder=2)

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

ax.plot()
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
pad = 0.03
ax.set_xlim(0.0 - pad, 5.0 + pad)
ax.set_ylim(-0.1 - pad, 1.1 + pad)
ax.set_aspect("equal")

plt.axis("off")

fname = os.path.join("img", "mesh-plot.pdf")
plt.savefig(fname, bbox_inches="tight")
plt.show()
