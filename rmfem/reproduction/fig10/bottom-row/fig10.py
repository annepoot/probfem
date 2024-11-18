import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_kappa(x, xi):
    theta = np.zeros_like(x)
    exp, sqrt, pi, sin = np.exp, np.sqrt, np.pi, np.sin

    for n, xi_i in enumerate(xi, 1):
        theta += sqrt(2) * xi_i / (n * pi) * sin(n * pi * x)

    kappa = exp(theta)
    return kappa


for std_noise in [1e-4, 1e-5, 1e-6]:
    x = np.linspace(0, 1, 1000)
    xi_true = [1.0, 1.0, 0.25, 0.25]
    kappa_true = get_kappa(x, xi_true)

    for n_elem in [10, 20, 40]:
        N_burn = 5000
        N_filter = 1000

        fname = "samples.csv"
        df = pd.read_csv(fname)
        df = df[abs(df["std_noise"] - std_noise) < 1e-20]
        df = df[df["n_elem"] == n_elem]
        df = df[(df["sample"] >= N_burn) & (df["sample"] % N_filter == 0)]

        fig, ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
        for i, sample in df.iterrows():
            xi = sample[["xi_1", "xi_2", "xi_3", "xi_4"]]
            kappa = get_kappa(x, xi)
            label = (
                r"$N = {}$".format(n_elem)
                if sample["sample"] == N_burn and sample["mesh"] == 0
                else None
            )
            ax.plot(x, kappa, color="gray", linewidth=1, alpha=0.5, label=label)
        ax.plot(x, kappa_true, color="black", label="Ref")
        ax.legend()
        ax.set_xlim((0, 1))
        ax.set_ylim((0.5, 2.5))
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$\kappa(x)$")
        # plt.savefig(fname="img/kappa-plot_N-{}_noise_{}".format(N, noise), dpi=600)
        plt.show()
