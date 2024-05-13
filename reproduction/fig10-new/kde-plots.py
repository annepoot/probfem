import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns


for noise in [1e-08, 1e-10, 1e-12, 1e-14, 1e-16]:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, tight_layout=True)
    plt.suptitle(r"KDE plots for $\sigma_\varepsilon$ = {:.0e}".format(np.sqrt(noise)))
    ax1.sharex(ax2)
    ax2.sharex(ax3)
    ax3.sharex(ax4)
    ax1.sharey(ax2)
    ax3.sharey(ax4)
    plt.setp((ax1, ax2, ax3, ax4), xlabel=r"$\xi_1$")
    ax1.set_ylabel(r"$\xi_3$")
    ax2.set_ylabel(r"$\xi_4$")
    ax3.set_ylabel(r"$\xi_5$")
    ax4.set_ylabel(r"$\xi_6$")

    handles = []
    labels = []

    for N in [10, 20, 40]:
        df = pd.read_csv("output/mcmc_xi_N-{}_noise-{}.csv".format(N, noise))
        df = df.transpose()
        df = df.rename(columns=lambda i: "xi_" + str(i + 1))
        df = df.rename(index=lambda s: int(s.split(".")[-1]) - 1)

        N_burn = 5000
        df = df.iloc[N_burn::10]

        sns.kdeplot(data=df, x="xi_1", y="xi_3", ax=ax1)
        sns.kdeplot(data=df, x="xi_1", y="xi_4", ax=ax2)
        sns.kdeplot(data=df, x="xi_1", y="xi_5", ax=ax3)
        sns.kdeplot(data=df, x="xi_1", y="xi_6", ax=ax4)

        pc1 = ax1.collections[-1]
        handles.append(Line2D([], [], color=pc1.get_edgecolor()[0]))
        labels.append(r"$N = {}$".format(N))

    handle = ax1.scatter([1.0], [0.25], color="k", zorder=2)
    ax2.scatter([1.0], [0.25], color="k", zorder=2)
    ax3.scatter([1.0], [1.0], color="k", zorder=2)
    ax4.scatter([1.0], [1.0], color="k", zorder=2)

    handles.append(handle)
    labels.append("Ref")

    ax2.legend(
        handles=handles, labels=labels, loc="center left", bbox_to_anchor=(1.0, 0.5)
    )
    plt.show()
