import os
import numpy as np
import pandas as pd

from util.io import read_csv_from


width = 0.10
N_burn = 10000
N_elem_range = [10, 20, 40]
folder = "4-observations"
ref = np.array([1.0, 1.0, 0.25, 0.25])


summary_dfs = []
for N_elem in N_elem_range:

    summary_subdfs = []
    for fem_type in ["fem", "rmfem", "rmfem-omit", "rmfem-2d"]:
        fname = os.path.join("output", folder, "samples-{}.csv".format(fem_type))
        df = read_csv_from(fname, "xi_1,xi_2,xi_3,xi_4")
        df = df[(df["sample"] >= N_burn)]
        df = df[df["n_elem"] == N_elem]
        df = df[["xi_1", "xi_2", "xi_3", "xi_4"]]

        mean = np.mean(df, axis=0)
        std = np.std(df, axis=0)
        error = abs(mean - ref)

        header_mean = "mean_" + fem_type
        header_std = "std_" + fem_type
        header_error = "error_" + fem_type

        summary_subdf = pd.DataFrame(
            {header_mean: mean, header_std: std, header_error: error}
        )
        summary_subdf["n_elem"] = N_elem

        summary_subdfs.append(summary_subdf)

    summary_df = pd.concat(summary_subdfs, axis=1)
    summary_dfs.append(summary_df)

summary = pd.concat(summary_dfs, axis=0)
n_elem_col = summary.pop("n_elem").iloc[:, 0]
summary = np.round(summary, 4)
summary.insert(0, "n_elem", n_elem_col)
summary.sort_index()
fname = os.path.join("img", folder, "summary.csv")
summary.to_csv(fname, index=False)
