import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def read_csv_from(fname, line, **kwargs):
    with open(fname) as f:
        pos = 0
        cur_line = f.readline()
        while not cur_line.startswith(line):
            pos = f.tell()
            cur_line = f.readline()
        f.seek(pos)
        return pd.read_csv(f, **kwargs)


def get_kappa(x, xi):
    theta = np.zeros_like(x)
    exp, sqrt, pi, sin = np.exp, np.sqrt, np.pi, np.sin

    for n, xi_i in enumerate(xi, 1):
        theta += sqrt(2) * xi_i / (n * pi) * sin(n * pi * x)

    kappa = exp(theta)
    return kappa


N_burn = 10000
N_filter = 100
folder = "4-observations"

x = np.linspace(0, 1, 1000)
xi_true = [1.0, 1.0, 0.25, 0.25]
kappa_true = get_kappa(x, xi_true)

for fem_type in ["fem", "rmfem"]:
    for N in [10, 20, 40]:
        c = {10: "C0", 20: "C1", 40: "C2"}[N]

        fname = os.path.join("output", folder, "samples-{}.csv".format(fem_type))
        df = read_csv_from(fname, "xi_1,xi_2,xi_3,xi_4")
        df = df[(df["sample"] >= N_burn) & (df["sample"] % N_filter == 0)]
        df = df[df["n_elem"] == N]
        df["n_elem"] = df["n_elem"].astype(str)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)

        for i, xi in df[["xi_1", "xi_2", "xi_3", "xi_4"]].iterrows():
            kappa = get_kappa(x, xi)
            label = r"$N = {}$".format(N) if i == df.index[0] else None
            ax.plot(x, kappa, color=c, linewidth=1, alpha=0.5, label=label)

        ax.plot(x, kappa_true, color="black", label="Ref")
        ax.set_xlim((0, 1))
        ax.set_ylim((0.5, 2.5))
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$\kappa$")
        ax.set_yticks([0.5, 1.0, 1.5, 2.0, 2.5])
        fname = "kappa-plot_{}_N-{}.pdf".format(fem_type, N)
        fname = os.path.join("img", folder, fname)
        plt.savefig(fname=fname, bbox_inches="tight")
        plt.show()
