import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
import seaborn as sns

from fem.meshing import calc_boundary_nodes, find_coords_in_nodeset
from fem.jive import CJiveRunner
from util.io import read_csv_from

from experiments.reproduction.inverse.three_point_hole.props import get_fem_props
from experiments.reproduction.inverse.three_point_hole.meshing import create_mesh


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


def is_on_outer_boundary(coord):
    tol = 1e-8

    if -tol < coord[0] < tol:
        return True
    elif -tol < coord[0] - 5.0 < tol:
        return True
    elif -tol < coord[1] < tol:
        return True
    elif -tol < coord[1] - 1.0 < tol:
        return True
    else:
        return False


def is_on_hole_boundary(coord, inode, boundary):
    if inode not in boundary:
        return False
    else:
        tol = 1e-8

        if coord[0] < tol:
            return False
        elif coord[0] > 5.0 - tol:
            return False
        elif coord[1] < tol:
            return False
        elif coord[1] > 1.0 - tol:
            return False
        else:
            return True


def plot_boundary(globdat, ax, **kws):
    elems = globdat["elemSet"]
    nodes = globdat["nodeSet"]
    dofs = globdat["dofSpace"]
    idofs = dofs.get_dofs(range(len(nodes)), ["dx", "dy"])
    du = globdat["state0"][idofs].reshape((-1, 2))

    boundary = calc_boundary_nodes(elems)

    for inode, coords in enumerate(nodes):
        if is_on_outer_boundary(coords):
            outer_start = inode
            break

    for inode, coords in enumerate(nodes):
        if is_on_hole_boundary(coords, inode, boundary):
            hole_start = inode
            break

    outer_loop = [outer_start]
    hole_loop = [hole_start]

    outer_loop_complete = False
    hole_loop_complete = False

    while not outer_loop_complete or not hole_loop_complete:
        for ielem, inodes in enumerate(elems):
            if not outer_loop_complete:
                idx = np.where(inodes == outer_loop[-1])[0]
                if len(idx) > 0:
                    next_inode = inodes[(idx[0] + 1) % len(inodes)]
                    next_coord = nodes[next_inode]
                    if is_on_outer_boundary(next_coord):
                        if next_inode not in outer_loop[-2:]:
                            outer_loop.append(next_inode)
                            if next_inode == outer_start:
                                outer_loop_complete = True

            if not hole_loop_complete:
                idx = np.where(inodes == hole_loop[-1])[0]
                if len(idx) > 0:
                    next_inode = inodes[(idx[0] + 1) % len(inodes)]
                    next_coord = nodes[next_inode]
                    if is_on_hole_boundary(next_coord, next_inode, boundary):
                        if next_inode not in hole_loop[-2:]:
                            hole_loop.append(next_inode)
                            if next_inode == hole_start:
                                hole_loop_complete = True

    for loop in [outer_loop, hole_loop]:
        coords = nodes[loop]
        disp = du[loop]
        xy = coords + scale * disp
        ax.plot(xy[:, 0], xy[:, 1], **kws)


def plot_observation_nodes(globdat, obs_locs, ax, **kws):
    inodes = find_coords_in_nodeset(obs_locs, globdat["nodeSet"])
    idofs = globdat["dofSpace"].get_dofs(inodes, ["dx", "dy"])
    disp_x = globdat["state0"][idofs[0::2]]
    disp_y = globdat["state0"][idofs[1::2]]
    xs = obs_locs[:, 0] + scale * disp_x
    ys = obs_locs[:, 1] + scale * disp_y
    ax.scatter(xs, ys, **kws)


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

std_corruption = 1e-4
h_range = [0.2, 0.1, 0.05]

for fem_type in ["fem"]:
    for i, h in enumerate(h_range):
        color = sns.color_palette("rocket_r", n_colors=8)[2 * i]

        fig, ax = plt.subplots()
        tol = 1e-8
        scale = 100

        ################
        # observations #
        ################
        df = read_csv_from("ground-truth.csv", "loc_x,loc_y")
        df = df[abs(np.fmod(df["loc_x"], 0.5)) < 1e-8]
        df = df[abs(np.fmod(df["loc_y"], 0.5)) < 1e-8]

        obs_locs = df[df["dof_type"] == "dx"][["loc_x", "loc_y"]].to_numpy()
        obs_disp_x = df[df["dof_type"] == "dx"]["value"].to_numpy()
        obs_disp_y = df[df["dof_type"] == "dy"]["value"].to_numpy()
        xs = obs_locs[:, 0] + scale * obs_disp_x
        ys = obs_locs[:, 1] + scale * obs_disp_y
        ax.scatter(xs, ys, marker=".", color="k", zorder=2)

        #############
        # MAP point #
        #############
        fname = "output/samples-{}.csv".format(fem_type)

        df = read_csv_from(fname, "x,y,a,theta,r_rel")
        df = df[df["sample"] >= N_start]
        df = df[df["sample"] <= N_end]
        df = df[df["sample"] % N_filter == 0]
        df = df[abs(df["std_corruption"] - std_corruption) < 1e-8]
        df["theta"] = np.fmod(df["theta"], 0.5 * np.pi)
        df = df[abs(df["h"] - h) < 1e-8]

        mean = np.mean(df, axis=0)

        nodes, elems = create_mesh(
            h=h,
            L=5.0,
            H=1.0,
            U=0.5,
            x=mean["x"],
            y=mean["y"],
            a=mean["a"],
            theta=mean["theta"],
            r_rel=mean["r_rel"],
            h_meas=0.5,
        )

        boundary = calc_boundary_nodes(elems)

        props = get_fem_props()
        jive = CJiveRunner(props, elems=elems)
        globdat = jive()

        plot_boundary(
            globdat,
            ax,
            color=color,
            linestyle="-",
            linewidth=1.0,
        )
        plot_observation_nodes(
            globdat,
            obs_locs,
            ax,
            color=color,
            marker="o",
            facecolors="none",
            linewidth=1.0,
        )

        ##############
        # true point #
        ##############
        nodes, elems = create_mesh(
            h=h,
            L=5.0,
            H=1.0,
            U=0.5,
            x=1.0,
            y=0.4,
            a=0.4,
            theta=np.pi / 6,
            r_rel=0.25,
            h_meas=0.5,
        )

        boundary = calc_boundary_nodes(elems)

        props = get_fem_props()
        jive = CJiveRunner(props, elems=elems)
        globdat = jive()

        plot_boundary(
            globdat,
            ax,
            color=color,
            linestyle=":",
            linewidth=1.0,
        )
        plot_observation_nodes(
            globdat,
            obs_locs,
            ax,
            color=color,
            marker="x",
            linewidth=1.0,
        )

        ax.set_aspect("equal")
        ax.set_xlim((0.0, 5.8))
        ax.set_ylim((-1.1, 1.4))
        plt.axis("off")

        fname = os.path.join("img", "boundary-plot_h-{:.2f}.pdf".format(h))
        plt.savefig(fname, bbox_inches="tight")
        plt.show()
