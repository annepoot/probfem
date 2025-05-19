import os
import numpy as np
import matplotlib.pyplot as plt

from experiments.inverse.frp_damage import params, misc

rve_size = params.geometry_params["rve_size"]
n_fiber = params.geometry_params["n_fiber"]
r_fiber = params.geometry_params["r_fiber"]
tol = params.geometry_params["tol_fiber"]
seed = params.geometry_params["seed_fiber"]

h = 0.02
noise = 1e-5

for k in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, np.inf]:
    E_matrix = params.material_params["E_matrix"]
    alpha = params.material_params["alpha"]
    beta = params.material_params["beta"]
    c = params.material_params["c"]
    d = params.material_params["d"]

    if np.isinf(k):
        fname = "posterior-samples_h-{:.3f}_noise-{:.0e}.npy"
        fname = os.path.join("output", fname.format(h, noise))
    else:
        fname = "posterior-samples_pod_h-{:.3f}_noise-{:.0e}_k-{}.npy"
        fname = os.path.join("output", fname.format(h, noise, k))

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

        fig, ax = plt.subplots()
        for i, sample in enumerate(samples):
            damage_sample = misc.sigmoid(sample, 1.0, 0.0)

            if i > 2000 and i % 10 == 0:
                alpha = 0.2 + 0.8 * i / len(samples)
                ax.plot(domain, damage_sample, color="C0", alpha=alpha, linewidth=0.1)

        ax.plot(domain, damage, color="k")
        title = r"Posterior samples, $k={}$"
        ax.set_title(title.format(k))
        ax.set_xlim((0, 0.2))
        ax.set_ylim((-0.1, 1.1))
        plt.show()
