import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime

from probability.sampling import MCMCRunner
from rwm_fem_props import get_rwm_fem_target
from cantilever_mesh import create_mesh

fem_type = "fem"

fname = "samples-{}.csv".format(fem_type)
current_time = datetime.now().strftime("%Y/%d/%m, %H:%M:%S")
n_sample = 100
std_corruption = 1e-3
h_range = [0.5, 0.2, 0.1]

file = open(fname, "w")

current_time = datetime.now().strftime("%Y/%d/%m, %H:%M:%S")
file.write("author = Anne Poot\n")
file.write(f"date, time = {current_time}\n")
file.write(f"n_sample = {n_sample}\n")
file.write(f"h = {h_range}\n")
file.write(f"std_corruption = fixed at {std_corruption}\n")

if fem_type == "fem":
    sigma_e = std_corruption
    recompute_logpdf = False
    file.write(f"sigma_e = fixed at {sigma_e}\n")

file.close()

for i, h in enumerate(h_range):

    if fem_type == "fem":
        sigma_e = std_corruption
        target = get_rwm_fem_target(
            h=h,
            std_corruption=std_corruption,
            sigma_e=sigma_e,
        )
        recompute_logpdf = False
    else:
        raise ValueError

    proposal = deepcopy(target.prior)
    start_value = target.prior.calc_mean()
    mcmc = MCMCRunner(
        target=target,
        proposal=proposal,
        n_sample=n_sample,
        start_value=start_value,
        seed=0,
        recompute_logpdf=False,
    )
    samples = mcmc()

    df = pd.DataFrame(samples, columns=["x", "a", "theta"])

    df["sample"] = df.index
    df["h"] = h
    df["std_corruption"] = std_corruption

    if fem_type == "fem":
        df["sigma_e"] = sigma_e
    else:
        raise ValueError

    write_header = h == h_range[0]
    df.to_csv(fname, mode="a", header=write_header, index=False)

create_mesh(
    lc=0.1,
    L=4,
    H=1,
    x=1,
    y=0.4,
    a=0.4,
    theta=np.pi / 6,
    r_rel=0.25,
    fname="cantilever.msh",
)
