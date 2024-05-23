import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_mesh(fname):
    nodes = []

    with open(fname, mode="r") as file:
        read = False
        for line in file:
            if "elements" in line:
                read = False

            if read:
                node = float(line.strip().split(" ")[1])
                nodes.append(node)

            if "nodes" in line:
                read = True

    return np.array(nodes)


def get_state0(fname):
    df = pd.read_csv(fname)
    last_col = df.columns[-1]
    print("state0 = {}".format(last_col))
    state0 = df[last_col]
    return state0.values


def get_files(mesh):
    meshfile = "output-before-fix/mesh-{}.mesh".format(mesh)
    state0file = "output-before-fix/mcmc_state0_N-5_noise-1e-20_mesh-{}.csv".format(
        mesh
    )
    return meshfile, state0file


plt.figure()

# Get perturbed solutions
for mesh in range(1, 21):
    meshfile, state0file = get_files(mesh)

    mesh = get_mesh(meshfile)
    state0 = get_state0(state0file)

    plt.plot(mesh, state0, color="gray", linewidth=0.5)

# Get reference solution
meshfile, state0file = get_files("ref")
mesh = get_mesh(meshfile)
state0 = get_state0(state0file)
plt.plot(mesh, state0, color="C0")

# Get exact solution
x = np.linspace(0, 1, 1000)
y = np.sin(2 * np.pi * x)
plt.plot(x, y, color="k")

# plt.savefig("img/state0_before-bug-fix.png", dpi=600)
plt.show()
