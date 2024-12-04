import numpy as np
import matplotlib.pyplot as plt

from myjivex.util.plotutils import create_dat
from probability.process import (
    GaussianProcess,
    InverseCovarianceOperator,
    NaturalCovarianceOperator,
    ProjectedPrior,
)
from bfem.observation import compute_bfem_observations
from fem.meshing import create_phi_from_globdat
from fem_props import get_fem_props

# Function to generate 1D meshes
def mesher(n, L=1, fname="bar"):
    dx = L / n
    if not "." in fname:
        fname += ".mesh"

    with open(fname, "w") as fmesh:
        fmesh.write("nodes (ID, x, [y], [z])\n")
        for i in range(n + 1):
            fmesh.write("%d %f\n" % (i, i * dx))
        fmesh.write("elements (node#1, node#2, [node#3, ...])\n")
        for i in range(n):
            fmesh.write("%d %d\n" % (i, i + 1))


cprops = get_fem_props("bar_coarse.mesh")
fprops = get_fem_props("bar_fine.mesh")

K_cov = InverseCovarianceOperator(fprops["model"])
M_cov = NaturalCovarianceOperator(fprops["model"], lumped_mass_matrix=False)

# Loop over different covariance matrices
for inf_cov in [M_cov, K_cov]:
    if isinstance(inf_cov, InverseCovarianceOperator):
        cov_name = "K"
    elif isinstance(inf_cov, NaturalCovarianceOperator):
        cov_name = "M"

    inf_prior = GaussianProcess(None, inf_cov)
    fine_prior = ProjectedPrior(inf_prior, fprops["init"], fprops["solve"])
    fglobdat = fine_prior.globdat

    # Loop over different densities of the coarse mesh
    for N_coarse in [4, 16, 64]:
        # Remesh the coarse mesh
        mesher(n=N_coarse, fname="bar_coarse.mesh")

        coarse_prior = ProjectedPrior(inf_prior, cprops["init"], cprops["solve"])
        cglobdat = coarse_prior.globdat

        PhiT = compute_bfem_observations(coarse_prior, fine_prior, fspace=False)
        H_obs = PhiT @ fglobdat["matrix0"]
        f_obs = PhiT @ fglobdat["extForce"]

        posterior = fine_prior.condition_on(H_obs, f_obs)
        mean_u_post = posterior.calc_mean()
        std_u_post = posterior.calc_std()

        u_coarse = cglobdat["state0"]
        u = fglobdat["state0"]
        Phi = create_phi_from_globdat(cglobdat, fglobdat)
        u_coarse = Phi @ u_coarse

        u_prior = fine_prior.calc_mean()
        u_post = posterior.calc_mean()
        std_u_prior = fine_prior.calc_std()
        std_u_post = posterior.calc_std()

        samples_u_post = posterior.calc_samples(20, 0)
        samples_u_prior = fine_prior.calc_samples(20, 0)

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
