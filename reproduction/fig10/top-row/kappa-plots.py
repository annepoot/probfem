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


for noise in [1e-08, 1e-10, 1e-12, 1e-14, 1e-16]:
    x = np.linspace(0, 1, 1000)
    xi_true = [1.0, 1.0, 0.25, 0.25]
    kappa_true = get_kappa(x, xi_true)

    for N in [10, 20, 40]:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)

        df = pd.read_csv("output/mcmc_xi_N-{}_noise-{}.csv".format(N, noise))
        df = df.transpose()
        df = df.rename(columns=lambda i: "xi_" + str(i + 1))
        df = df.rename(index=lambda s: int(s.split(".")[-1]) - 1)

        N_burn = 5000
        df = df.iloc[N_burn::10]

        for i, xi in df.iterrows():
            kappa = get_kappa(x, xi)
            label = r"$N = {}$".format(N) if i == N_burn else None
            ax.plot(x, kappa, color="gray", linewidth=1, alpha=0.5, label=label)
        ax.plot(x, kappa_true, color="black", label="Ref")
        ax.legend()
        ax.set_xlim((0, 1))
        ax.set_ylim((0.5, 2.5))
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$\kappa(x)$")
        plt.savefig(fname="img/kappa-plot_N-{}_noise_{}".format(N, noise), dpi=600)
        plt.show()
