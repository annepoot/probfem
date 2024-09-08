import numpy as np
import matplotlib.pyplot as plt

from myjive.app import main
import myjive.util.proputils as pu
from myjivex import declare_all as declarex
from bfem import declare_all as declarebfem
from myjivex.util.plotutils import create_dat


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


# Load the properties of the run
props = pu.parse_file("tapered.pro")

# Loop over different covariance matrices
for covariance in ["K", "M"]:
    props["model"]["bfem"]["prior"]["latent"]["cov"] = covariance

    # Loop over different densities of the coarse mesh
    for N_coarse in [2, 4, 8, 16, 32, 64]:

        # Remesh the coarse mesh
        mesher(n=N_coarse, fname="bar_coarse.mesh")

        # Now do the actual run
        extra_declares = [declarex, declarebfem]
        globdat = main.jive(props, extra_declares=extra_declares)

        # Save the fine and projected coarse displacements
        u_coarse = globdat["coarse"]["state0"]
        u = globdat["fine"]["state0"]
        Phi = globdat["Phi"]
        u_coarse = Phi @ u_coarse

        # Get the prior and posterior means and standard deviations
        u_prior = globdat["gp"]["mean"]["prior"]["state0"]
        u_post = globdat["gp"]["mean"]["posterior"]["state0"]
        std_u_prior = globdat["gp"]["std"]["prior"]["state0"]
        std_u_post = globdat["gp"]["std"]["posterior"]["state0"]

        # Get the prior and posterior samples
        samples_u_prior = globdat["gp"]["samples"]["prior"]["state0"]
        samples_u_post = globdat["gp"]["samples"]["posterior"]["state0"]

        # Use a fine linspace for plotting
        x = np.linspace(0, 1, len(u))

        # Create figure 1 directly using matplotlib
        plt.figure()
        plt.plot(x, u_post, label="posterior mean")
        plt.plot(x, u_prior, label="prior mean")
        plt.plot(x, samples_u_post, color="gray", linewidth=0.2)
        plt.plot(x, samples_u_prior, color="gray", linewidth=0.2)
        plt.fill_between(x, u_post - 2 * std_u_post, u_post + 2 * std_u_post, alpha=0.3)
        plt.fill_between(
            x, u_prior - 2 * std_u_prior, u_prior + 2 * std_u_prior, alpha=0.3
        )
        plt.plot(x, u_coarse, label="coarse solution")
        plt.plot(x, u, label="fine solution")
        plt.ylim((-13, 17))
        plt.legend(loc="upper left")
        plt.title(r"${}$-prior, $n_c={}$, $n_f=64$".format(covariance, N_coarse))
        plt.show()

        # Create output files for latex
        create_dat(data=x, headers="x", fname=covariance + "/mesh_Nf-64.dat")

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
            fname=covariance + "/results_Nc-{}_Nf-64.dat".format(N_coarse),
        )

        create_dat(
            data=samples_u_prior,
            headers="prior_sample_{}",
            fname=covariance + "/samples-prior_Nc-{}_Nf-64.dat".format(N_coarse),
        )

        create_dat(
            data=samples_u_post,
            headers="posterior_sample_{}",
            fname=covariance + "/samples-posterior_Nc-{}_Nf-64.dat".format(N_coarse),
        )
