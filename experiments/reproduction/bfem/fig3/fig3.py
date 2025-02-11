import numpy as np
import matplotlib.pyplot as plt

from myjive.solver import Constrainer
from myjivex.util.plotutils import create_dat
from probability.process import (
    GaussianProcess,
    InverseCovarianceOperator,
    NaturalCovarianceOperator,
    ProjectedPrior,
)
from bfem.observation import compute_bfem_observations
from fem.meshing import create_phi_from_globdat, mesh_interval_with_line2
from fem.jive import CJiveRunner
from util.linalg import Matrix

from experiments.reproduction.bfem.fig3.fem_props import get_fem_props


_, ref_elems = mesh_interval_with_line2(n=64)

ref_module_props = get_fem_props()
ref_model_props = ref_module_props.pop("model")
ref_jive_runner = CJiveRunner(ref_module_props, elems=ref_elems)

K_cov = InverseCovarianceOperator(
    model_props=ref_model_props,
    scale=1.0,
)
M_cov = NaturalCovarianceOperator(
    model_props=ref_model_props,
    scale=1.0,
    lumped_mass_matrix=False,
)

# Loop over different covariance matrices
for inf_cov in [M_cov, K_cov]:
    if isinstance(inf_cov, InverseCovarianceOperator):
        cov_name = "K"
    elif isinstance(inf_cov, NaturalCovarianceOperator):
        cov_name = "M"

    inf_prior = GaussianProcess(None, inf_cov)
    ref_jive_runner = CJiveRunner(ref_module_props, elems=ref_elems)
    ref_prior = ProjectedPrior(prior=inf_prior, jive_runner=ref_jive_runner)
    ref_globdat = ref_prior.globdat

    # Loop over different densities of the coarse mesh
    for N_coarse in [4, 16, 64]:
        _, obs_elems = mesh_interval_with_line2(n=N_coarse)

        obs_module_props = get_fem_props()
        obs_model_props = obs_module_props.pop("model")
        obs_jive_runner = CJiveRunner(obs_module_props, elems=obs_elems)
        obs_prior = ProjectedPrior(prior=inf_prior, jive_runner=obs_jive_runner)
        obs_globdat = obs_prior.globdat

        PhiT = compute_bfem_observations(obs_prior, ref_prior, fspace=False)

        K = ref_globdat["matrix0"]
        c = ref_globdat["constraints"]
        Kc = Constrainer(c, K).get_output_matrix()

        Phi = Matrix(PhiT.T, name="Phi")
        Kc = Matrix(Kc, name="Kc")

        H_obs = Phi.T @ Kc
        f_obs = PhiT @ ref_globdat["extForce"]

        u_coarse = obs_globdat["state0"]
        u = ref_globdat["state0"]
        Phi = create_phi_from_globdat(obs_globdat, ref_globdat)
        u_coarse = Phi @ u_coarse

        posterior = ref_prior.condition_on(H_obs, f_obs)

        samples_u_prior = ref_prior.calc_samples(50, 0)
        samples_u_post = posterior.calc_samples(50, 0)

        u_prior = ref_prior.calc_mean()
        u_post = posterior.calc_mean()
        std_u_prior = ref_prior.calc_std()
        std_u_post = posterior.calc_std()
        # std_u_prior = np.std(samples_u_prior, axis=0)
        # std_u_post = np.std(samples_u_post, axis=0)

        # Use a fine linspace for plotting
        x = np.linspace(0, 1, len(u))

        # Create figure 1 directly using matplotlib
        plt.figure()
        plt.plot(x, u_post, color="C0", label="posterior mean")
        plt.plot(x, u_prior, color="C1", label="prior mean")
        plt.plot(x, samples_u_post.T, color="C0", linewidth=0.2)
        plt.plot(x, samples_u_prior.T, color="C1", linewidth=0.2)
        plt.fill_between(x, u_post - 2 * std_u_post, u_post + 2 * std_u_post, alpha=0.3)
        plt.fill_between(
            x, u_prior - 2 * std_u_prior, u_prior + 2 * std_u_prior, alpha=0.3
        )
        plt.plot(x, u_coarse, color="C2", label="coarse solution")
        plt.plot(x, u, color="C3", label="fine solution")
        plt.ylim((-13, 17))
        plt.legend(loc="upper left")
        plt.show()

        # Create output files for latex
        create_dat(data=x, headers="x", fname=cov_name + "/mesh_Nf-64.dat")

        create_dat(
            data=[u_prior, u_post, std_u_prior, std_u_post, u_coarse, u],
            headers=[
                "u_prior",
                "u_posterior",
                "std_u_prior",
                "std_u_posterior",
                "u_coarse",
                "u_fine",
            ],
            fname=cov_name + "/results_Nc-{}_Nf-64.dat".format(N_coarse),
        )

        create_dat(
            data=samples_u_prior,
            headers="prior_sample_{}",
            fname=cov_name + "/samples-prior_Nc-{}_Nf-64.dat".format(N_coarse),
        )

        create_dat(
            data=samples_u_post,
            headers="posterior_sample_{}",
            fname=cov_name + "/samples-posterior_Nc-{}_Nf-64.dat".format(N_coarse),
        )
