import pandas as pd
import seaborn as sns

variables = ["xi_1", "xi_2", "xi_3", "xi_4"]
refs_by_var = {
    "xi_1": 1.0,
    "xi_2": 1.0,
    "xi_3": 0.25,
    "xi_4": 0.25,
}
width = 0.05
lims_by_var = {}
for var in variables:
    ref = refs_by_var[var]
    lims_by_var[var] = (ref - width, ref + width)
labels_by_var = {
    "xi_1": r"$\xi_1$",
    "xi_2": r"$\xi_2$",
    "xi_3": r"$\xi_3$",
    "xi_4": r"$\xi_4$",
}

for noise in [1e-08, 1e-10, 1e-12, 1e-14, 1e-16]:
    N_burn = 5000
    df_list = []

    for N in [10, 20, 40]:
        sub_df = pd.read_csv("output/mcmc_xi_N-{}_noise-{}.csv".format(N, noise))
        sub_df = sub_df.transpose()
        sub_df = sub_df.rename(columns=lambda i: "xi_" + str(i + 1))
        sub_df = sub_df.rename(index=lambda s: int(s.split(".")[-1]) - 1)
        sub_df = sub_df.iloc[N_burn::10]
        sub_df["N"] = str(N)
        df_list.append(sub_df)

    df = pd.concat(df_list, axis=0)

    grid = sns.PairGrid(data=df, vars=variables, hue="N", diag_sharey=False)
    grid.map_upper(sns.scatterplot)
    grid.map_lower(sns.kdeplot)
    grid.map_diag(sns.kdeplot)

    nvar = len(variables)
    for i, var in enumerate(variables):
        lims = lims_by_var[var]
        grid.axes[(i + 1) % nvar, i].set_xlim(lims)
        grid.axes[i, (i + 1) % nvar].set_ylim(lims)

    for i, xvar in enumerate(variables):
        for j, yvar in enumerate(variables):
            xlabel = labels_by_var[xvar]
            ylabel = labels_by_var[yvar]
            grid.axes[j, i].xaxis.set_label_text(xlabel)
            grid.axes[j, i].yaxis.set_label_text(ylabel)

            xref = refs_by_var[xvar]
            yref = refs_by_var[yvar]
            if i == j:
                grid.axes[j, i].axvline(x=xref, color="k", label="ref", zorder=2)
            else:
                grid.axes[j, i].scatter(
                    [xref], [yref], color="k", label="ref", zorder=2
                )

    grid.add_legend()
    grid.savefig(fname="img/pairgrid-plot_noise-{}".format(noise), dpi=600)
