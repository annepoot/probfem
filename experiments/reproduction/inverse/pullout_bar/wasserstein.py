import numpy as np
import ot

from util.io import read_csv_from


N_filter = 50
N_start = 10000
N_end = 20000
N_sample = (N_end - N_start) // N_filter + 1
std_corruption = 1e-4

n_elem_ref = 128
fem_type_ref = "fem"

fname_ref = "output/samples-{}.csv".format(fem_type_ref)
df_ref = read_csv_from(fname_ref, "log_E,log_k")
df_ref = df_ref[df_ref["sample"] >= N_start]
df_ref = df_ref[df_ref["sample"] <= N_end]
df_ref = df_ref[df_ref["sample"] % N_filter == 0]
df_ref = df_ref[df_ref["n_elem"] == n_elem_ref]

df_ref = df_ref[["log_E", "log_k"]]
mins, maxs = np.min(df_ref, axis=0), np.max(df_ref, axis=0)
df_ref = (df_ref - mins) / (maxs - mins)

print("fem_type  \th    \tdistance (mean +- std)")
for fem_type in ["fem", "bfem", "rmfem", "statfem"]:
    for n_elem in [1, 2, 4, 8, 16, 32, 64]:
        Ws = []

        for seed in range(20):
            fname = "output/samples-{}_seed-{}.csv".format(fem_type, seed)
            df = read_csv_from(fname, "log_E,log_k")
            df = df[df["sample"] >= N_start]
            df = df[df["sample"] <= N_end]
            df = df[df["sample"] % N_filter == 0]
            df = df[df["n_elem"] == n_elem]

            df = df[["log_E", "log_k"]]
            df = (df - mins) / (maxs - mins)

            weights = np.ones(N_sample) / N_sample

            M = ot.dist(df.to_numpy(), df_ref.to_numpy(), metric="euclid") ** 2
            G = ot.emd(weights, weights, M)
            W = np.sqrt(np.sum(G * M))

            Ws.append(W)

        W_mean = np.mean(Ws)
        W_std = np.std(Ws)
        print(
            "{:10s}\t{:.3f}\t{:.6f} +- {:.6f}".format(fem_type, n_elem, W_mean, W_std)
        )
