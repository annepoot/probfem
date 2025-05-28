import os
import numpy as np
import matplotlib.pyplot as plt

from probability.process import GaussianProcess, ZeroMeanFunction, SquaredExponential

from experiments.inverse.frp_damage import params, misc

rve_size = params.geometry_params["rve_size"]
n_fiber = params.geometry_params["n_fiber"]
r_fiber = params.geometry_params["r_fiber"]
tol = params.geometry_params["tol_fiber"]
seed = params.geometry_params["seed_fiber"]

h = 0.02
noise = 1e-5
l = 10
fem_type = "pod"

for k in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, np.inf]:
    E_matrix = params.material_params["E_matrix"]
    alpha = params.material_params["alpha"]
    beta = params.material_params["beta"]
    c = params.material_params["c"]
    d = params.material_params["d"]

    if np.isinf(k):
        fname = "posterior-samples_h-{:.3f}_noise-{:.0e}.npy"
        fname = os.path.join("output", fname.format(h, noise))
    elif fem_type == "pod":
        fname = "posterior-samples_pod_h-{:.3f}_noise-{:.0e}_k-{}.npy"
        fname = os.path.join("output-grid", fname.format(h, noise, k))
    elif fem_type == "bpod":
        fname = "posterior-samples_bpod_h-{:.3f}_noise-{:.0e}_k-{}_l-{}.npy"
        fname = os.path.join("output-grid", fname.format(h, noise, k, l))
    else:
        raise ValueError

    try:
        samples = np.load(fname)
        samples_found = True
    except:
        samples_found = False

    if samples_found:
        domain = np.linspace(0.0, 0.2, 101)
        input_map = (len(domain) - 1) / np.max(domain)
        saturation = misc.saturation(domain, alpha, beta, c)
        damage = misc.damage(saturation, d)

        target = GaussianProcess(
            mean=ZeroMeanFunction(),
            cov=SquaredExponential(l=0.02, sigma=2.0),
        )

        U, s, _ = np.linalg.svd(target.calc_cov(domain, domain))

        trunc = 10
        eigenfuncs = U[:, :trunc]
        eigenvalues = s[:trunc]

        fig, ax = plt.subplots()
        for i, sample in enumerate(samples):
            if len(sample) == 10:
                damage_sample = misc.sigmoid(eigenfuncs @ sample, 1.0, 0.0)
            elif len(sample) == 101:
                damage_sample = misc.sigmoid(sample, 1.0, 0.0)
            else:
                assert False

            if i > 10000 and i % 100 == 0:
                ax.plot(domain, damage_sample, color="C0", linewidth=0.1)

        ax.plot(domain, damage, color="k")

        if fem_type == "pod":
            ax.set_title(r"Posterior samples POD, $k={}$".format(k))
        if fem_type == "bpod":
            ax.set_title(r"Posterior samples BPOD, $k={}$, $l={}$".format(k, l))
        else:
            assert False

        ax.set_xlim((0, 0.2))
        ax.set_ylim((-0.1, 1.1))
        ax.set_xlabel(r"Distance to fiber")
        ax.set_ylabel(r"Damage")
        ax.set_xticks([0.00, 0.05, 0.10, 0.15, 0.20])
        ax.set_yticks([0.0, 0.5, 1.0])
        plt.show()
