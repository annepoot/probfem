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
    fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, tight_layout=True, figsize=(8, 8))
    plt.suptitle(
        r"$\kappa$ as a function of $x$ for $\sigma_\varepsilon$ = {:.0e}".format(
            np.sqrt(noise)
        )
    )
    plt.setp(axs, xlabel=r"$x$",  ylabel=r"$\kappa$")

    x = np.linspace(0, 1, 1000)
    xi_true = [1.0, 1.0, 0.25, 0.25, 1.0, 1.0]
    kappa_true = get_kappa(x, xi_true)

    for N, ax in zip([10, 20, 40], axs):
        df = pd.read_csv("output/mcmc_xi_N-{}_noise-{}.csv".format(N, noise))
        df = df.transpose()
        df = df.rename(columns=lambda i: "xi_" + str(i + 1))
        df = df.rename(index=lambda s: int(s.split(".")[-1]) - 1)

        N_burn = 5000
        df = df.iloc[N_burn::10]

        for i, xi in df.iterrows():
            kappa = get_kappa(x, xi)
            label = r"$N = {}$".format(N) if i == N_burn else None
            ax.plot(x, kappa, color="gray", linewidth=0.1, label=label)
        ax.plot(x, kappa_true, color="black", label="Ref")
        ax.legend()
    plt.show()
