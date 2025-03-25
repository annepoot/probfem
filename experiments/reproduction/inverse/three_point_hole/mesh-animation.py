import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
from matplotlib.animation import FuncAnimation

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

for fem_type in ["fem", "bfem", "rmfem", "statfem"]:
    for h in [0.20, 0.10, 0.05]:
        fname = os.path.join("output", "samples-{}.csv".format(fem_type))

        N_filter = 10
        N_start = 0
        N_end = 20000
        std_corruption = 1e-4

        df = read_csv_from(fname, "x,y,a,theta,r_rel")
        df = df[df["sample"] >= N_start]
        df = df[df["sample"] <= N_end]
        df = df[df["sample"] % N_filter == 0]
        df = df[abs(df["std_corruption"] - std_corruption) < 1e-8]
        df = df[abs(df["h"] - h) < 1e-8]
        df["theta"] = np.fmod(df["theta"], 0.5 * np.pi)

        frames = []

        # Create figure and axis
        fig, ax = plt.subplots()
        plt.axis("off")

        # Set limits and aspect ratio
        pad = 0.03
        ax.set_xlim(0.0 - pad, 5.0 + pad)
        ax.set_ylim(-0.1 - pad, 1.1 + pad)
        ax.set_aspect("equal")

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

        true_patch = get_hole_patch(1.0, 0.4, 0.4, np.pi / 6, 0.1)
        true_patch.set_edgecolor("0.7")
        true_patch.set_alpha(1.0)
        true_patch.set_linewidth(1.0)
        true_patch.set_linestyle("dashed")
        ax.add_patch(true_patch)

        plot_edges = []
        hole = None

        def update(frame):
            global plot_edges, hole

            print("frame", frame)
            row = df.iloc[frame]
            x, y, a, theta, r = row[["x", "y", "a", "theta", "r"]]

            nodes, elems = create_mesh(
                h=h,
                L=5.0,
                H=1.0,
                U=0.5,
                x=x,
                y=y,
                a=a,
                theta=theta,
                r_rel=max(r, 1e-3),
                h_meas=0.5,
            )

            new_edges = []
            for inodes in elems:
                for i in range(len(inodes)):
                    new_edges.append(inodes[[i, (i + 1) % 3]])
            new_edges = np.unique(np.sort(np.array(new_edges), axis=1), axis=0)

            for i in range(min(len(new_edges), len(plot_edges))):
                new_edge = new_edges[i]
                coords = nodes[new_edge]
                plot_edges[i].set_data(coords[:, 0], coords[:, 1])

            if len(new_edges) > len(plot_edges):
                for i in range(len(plot_edges), len(new_edges)):
                    new_edge = new_edges[i]
                    coords = nodes[new_edge]
                    plot_edge = ax.plot(
                        coords[:, 0], coords[:, 1], color="0.7", linewidth=0.5, zorder=0
                    )
                    plot_edges.append(plot_edge[0])
            elif len(plot_edges) > len(new_edges):
                for i in range(len(new_edges), len(plot_edges)):
                    plot_edge = plot_edges[i]
                    plot_edge.set_data([], [])

            if hole is not None:
                hole.remove()

            ref_patch = get_hole_patch(x, y, a, theta, r)
            ref_patch.set_edgecolor("black")
            ref_patch.set_alpha(1.0)
            ref_patch.set_linewidth(1.0)
            hole = ax.add_patch(ref_patch)

            return hole, plot_edges

        # Create the animation
        ani = FuncAnimation(fig, update, frames=2000, interval=1000 / 30)
        fname = "mesh-animation_{}_h-{:.2f}.mp4".format(fem_type, h)
        fname = os.path.join("ani", fname)
        ani.save(fname, writer="ffmpeg")
        plt.show()
