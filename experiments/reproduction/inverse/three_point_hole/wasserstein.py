import numpy as np
import ot

from util.io import read_csv_from


N_filter = 50
N_start = 10000
N_end = 20000
N_sample = (N_end - N_start) // N_filter + 1
std_corruption = 1e-4

h_ref = 0.01
fem_type_ref = "fem"

fname_ref = "output/samples-{}.csv".format(fem_type_ref)
df_ref = read_csv_from(fname_ref, "x,y,a,theta,r_rel")
df_ref = df_ref[df_ref["sample"] >= N_start]
df_ref = df_ref[df_ref["sample"] <= N_end]
df_ref = df_ref[df_ref["sample"] % N_filter == 0]
df_ref = df_ref[abs(df_ref["std_corruption"] - std_corruption) < 1e-8]
df_ref["theta"] = np.fmod(df_ref["theta"], 0.5 * np.pi)
df_ref = df_ref[abs(df_ref["h"] - h_ref) < 1e-8]


def build_cost_matrix(df, df_ref):
    df_main = df[["x", "y", "a", "r_rel"]]
    df_ref_main = df_ref[["x", "y", "a", "r_rel"]]
    df_theta = df[["theta"]]
    df_ref_theta = df_ref[["theta"]]

    min_main, max_main = np.min(df_ref_main, axis=0), np.max(df_ref_main, axis=0)
    min_angle, max_angle = 0.0, 0.5 * np.pi

    df_main = (df_main - min_main) / (max_main - min_main)
    df_ref_main = (df_ref_main - min_main) / (max_main - min_main)
    df_theta = (df_theta - min_angle) / (max_angle - min_angle)
    df_ref_theta = (df_ref_theta - min_angle) / (max_angle - min_angle)

    M_main = ot.dist(df_main.to_numpy(), df_ref_main.to_numpy(), metric="euclidean")
    M_angle = df_theta.to_numpy() - df_ref_theta.to_numpy().T
    M_angle = np.abs((M_angle + 0.5) % 1.0 - 0.5)

    return M_main**2 + M_angle**2


print("fem_type  \th    \tdistance (mean +- std)")
for fem_type in ["fem", "bfem", "rmfem", "statfem"]:
    for h in [0.20, 0.10, 0.05]:
        Ws = []

        for seed in range(1):
            # fname = "output/samples-{}_seed-{}.csv".format(fem_type, seed)
            fname = "output/samples-{}.csv".format(fem_type)
            df = read_csv_from(fname, "x,y,a,theta,r_rel")
            df = df[df["sample"] >= N_start]
            df = df[df["sample"] <= N_end]
            df = df[df["sample"] % N_filter == 0]
            df = df[abs(df["std_corruption"] - std_corruption) < 1e-8]
            df["theta"] = np.fmod(df["theta"], 0.5 * np.pi)
            df = df[abs(df["h"] - h) < 1e-8]

            weights = np.ones(N_sample) / N_sample

            M = build_cost_matrix(df, df_ref)
            G = ot.emd(weights, weights, M)
            W = np.sqrt(np.sum(G * M))

            Ws.append(W)

        W_mean = np.mean(Ws)
        W_std = np.std(Ws)
        print("{:10s}\t{:.3f}\t{:.6f} +- {:.6f}".format(fem_type, h, W_mean, W_std))
