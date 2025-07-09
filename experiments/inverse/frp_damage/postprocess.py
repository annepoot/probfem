import os
import numpy as np
from scipy.sparse import diags_array
import matplotlib as mpl
import matplotlib.pyplot as plt

from probability.multivariate import Gaussian, SymbolicCovariance
from probability.process import GaussianProcess, ZeroMeanFunction, SquaredExponential
from util.linalg import Matrix

from experiments.inverse.frp_damage import params, misc

rve_size = params.geometry_params["rve_size"]
n_fiber = params.geometry_params["n_fiber"]
r_fiber = params.geometry_params["r_fiber"]
tol = params.geometry_params["tol_fiber"]
seed = params.geometry_params["seed_fiber"]

h = 0.010
sigma_e = 1e-3
k = 1
l = 10
seed = 0
n_burn = 10000
fem_type = "fem"

for h in [0.100, 0.050, 0.020, 0.010]:
    E_matrix = params.material_params["E_matrix"]
    alpha = params.material_params["alpha"]
    beta = params.material_params["beta"]
    c = params.material_params["c"]
    d = params.material_params["d"]

    sample_list = []
    logpdf_list = []

    for seed in range(10):
        if np.isinf(k):
            fname = "posterior-samples_fem_h-{:.3f}_noise-{:.0e}_seed-{}.npy"
            fname = fname.format(h, sigma_e, seed)
            fname = os.path.join("output", "fem", fname)
        elif fem_type == "pod":
            fname = "posterior-samples_pod_h-{:.3f}_noise-{:.0e}_k-{}_seed-{}.npy"
            fname = fname.format(h, sigma_e, k, seed)
            fname = os.path.join("output", "pod", fname)
            n_filter = 1000
        elif fem_type == "bpod":
            fname = "posterior-samples_bpod_h-{:.3f}_noise-{:.0e}_k-{}_l-{}_seed-{}.npy"
            fname = fname.format(h, sigma_e, k, l, seed)
            fname = os.path.join("output", "bpod", fname)
            n_filter = 1000
        elif fem_type == "fem":
            fname = "posterior-samples_fem_h-{:.3f}_noise-{:.0e}_seed-{}.npy"
            fname = fname.format(h, sigma_e, seed)
            fname = os.path.join("output", "fem", fname)
            n_filter = 1000
        elif fem_type == "bfem":
            fname = "posterior-samples_bfem_h-{:.3f}_noise-{:.0e}_seed-{}.npy"
            fname = fname.format(h, sigma_e, seed)
            fname = os.path.join("output", "bfem", fname)
            n_filter = 1000
        elif fem_type == "bfem-hier":
            fname = "posterior-samples_bfem-hier_h-{:.3f}_noise-{:.0e}_seed-{}.npy"
            fname = fname.format(h, sigma_e, seed)
            fname = os.path.join("output", "bfem", fname)
            n_filter = 1000
        elif fem_type == "bfem-heter":
            fname = "posterior-samples_bfem-heter_h-{:.3f}_noise-{:.0e}_seed-{}.npy"
            fname = fname.format(h, sigma_e, seed)
            fname = os.path.join("output", "bfem", fname)
            n_filter = 1000
        else:
            raise ValueError

        try:
            samples = np.load(fname)
            samples_found = True
        except:
            samples_found = False

        if samples_found:
            sample_list.append(samples[n_burn:])

    if len(sample_list) > 0 or k == 0:
        domain = np.linspace(0.0, 0.2, 101)
        input_map = (len(domain) - 1) / np.max(domain)
        saturation = misc.saturation(domain, alpha, beta, c)
        true_damage = misc.damage(saturation, d) * 100

        target = GaussianProcess(
            mean=ZeroMeanFunction(),
            cov=SquaredExponential(l=0.02, sigma=2.0),
        )

        U, s, _ = np.linalg.svd(target.calc_cov(domain, domain))

        trunc = 10
        eigenfuncs = U[:, :trunc]
        eigenvalues = s[:trunc]

        if k == 0:
            kl_cov = SymbolicCovariance(Matrix(diags_array(eigenvalues), name="S"))
            kl_target = Gaussian(mean=None, cov=kl_cov)
            samples = kl_target.calc_samples(100, 0)
            n_filter = 1
        else:
            samples = np.concatenate(sample_list)

        color = mpl.colormaps["viridis"](0.5)

        fig, ax = plt.subplots()
        for i, sample in enumerate(samples):
            if len(sample) == 10:
                damage_sample = misc.sigmoid(eigenfuncs @ sample, 1.0, 0.0) * 100
            elif len(sample) == 101:
                damage_sample = misc.sigmoid(sample, 1.0, 0.0) * 100
            else:
                assert False

            if i % n_filter == 0:
                ax.plot(domain, damage_sample, color=color, linewidth=0.3)

        ax.plot(domain, true_damage, color="k", linestyle="--")

        if fem_type == "pod":
            title = r"Posterior samples POD, $k={}$".format(k)
            fname = "posterior-samples_pod_h-{:.3f}_noise-{:.0e}_k-{}.pdf"
            fname = os.path.join("img", fname.format(h, sigma_e, k))
        elif "bpod" in fem_type:
            title = r"Posterior samples BPOD, $k={}$, $l={}$".format(k, l)
            fname = "posterior-samples_bpod_h-{:.3f}_noise-{:.0e}_k-{}_l-{}.pdf"
            fname = os.path.join("img", fname.format(h, sigma_e, k, l))
        elif fem_type == "fem":
            title = r"Posterior samples FEM, $h={:.3f}$".format(h)
            fname = "posterior-samples_fem_h-{:.3f}_noise-{:.0e}.pdf"
            fname = os.path.join("img", fname.format(h, sigma_e))
        elif fem_type == "bfem":
            title = r"Posterior samples BFEM, $h={:.3f}$".format(h)
            fname = "posterior-samples_bfem_h-{:.3f}_noise-{:.0e}.pdf"
            fname = os.path.join("img", fname.format(h, sigma_e))
        elif fem_type == "bfem-hier":
            title = r"Posterior samples BFEM (hierarchical), $h={:.3f}$".format(h)
            fname = "posterior-samples_bfem-hier_h-{:.3f}_noise-{:.0e}.pdf"
            fname = os.path.join("img", fname.format(h, sigma_e))
        elif fem_type == "bfem-heter":
            title = r"Posterior samples BFEM (nonhierarchical), $h={:.3f}$".format(h)
            fname = "posterior-samples_bfem-heter_h-{:.3f}_noise-{:.0e}.pdf"
            fname = os.path.join("img", fname.format(h, sigma_e))
        else:
            assert False

        ax.set_xlim((0, 0.2))
        ax.set_ylim((0, 100))
        ax.set_xlabel(r"distance to fiber (mm)")
        ax.set_ylabel(r"stiffness reduction (\%)")
        ax.set_xticks([0.00, 0.05, 0.10, 0.15, 0.20])
        ax.set_yticks([0, 25, 50, 75, 100])
        plt.savefig(fname, bbox_inches="tight")
        plt.show()
