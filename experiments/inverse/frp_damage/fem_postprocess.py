import numpy as np
import matplotlib.pyplot as plt

from experiments.inverse.frp_damage import params, misc

rve_size = params.geometry_params["rve_size"]
n_fiber = params.geometry_params["n_fiber"]
r_fiber = params.geometry_params["r_fiber"]
tol = params.geometry_params["tol_fiber"]
seed = params.geometry_params["seed_fiber"]

for h in [0.05, 0.02, 0.01]:
    for noise in [1e-5]:
        E_matrix = params.material_params["E_matrix"]
        alpha = params.material_params["alpha"]
        beta = params.material_params["beta"]
        c = params.material_params["c"]
        d = params.material_params["d"]

        fname = "output/posterior-samples_h-{:.3f}_noise-{:.0e}.npy".format(h, noise)
        samples = np.load(fname)

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
        title = r"Posterior samples, $h={}$, $\sigma_e = 10^{{{}}}$"
        ax.set_title(title.format(h, int(np.log10(noise))))
        ax.set_xlim((0, 0.2))
        ax.set_ylim((-0.1, 1.1))
        ax.set_xlabel(r"Distance to fiber")
        ax.set_ylabel(r"Damage")
        ax.set_xticks([0.00, 0.05, 0.10, 0.15, 0.20])
        ax.set_yticks([0.0, 0.5, 1.0])
        plt.show()
