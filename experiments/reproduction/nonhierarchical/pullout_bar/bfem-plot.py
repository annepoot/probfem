import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from experiments.reproduction.nonhierarchical.pullout_bar.props import get_fem_props
from fem.jive import CJiveRunner
from fem.meshing import mesh_interval_with_line2, create_phi_from_globdat
from probability.process import (
    GaussianProcess,
    InverseCovarianceOperator,
    ProjectedPrior,
)
from bfem.observation import compute_bfem_observations

props = get_fem_props()


def u_exact(x):
    k = props["model"]["model"]["spring"]["k"]
    E = props["model"]["model"]["elastic"]["material"]["E"]
    f = props["model"]["model"]["neum"]["initLoad"]

    nu = np.sqrt(k / E)
    eps = f / E

    A = eps / (nu * (np.exp(nu) - np.exp(-nu)))

    return A * (np.exp(nu * x) + np.exp(-nu * x))


n_elem = 4
obs_nodes, obs_elems = mesh_interval_with_line2(n=n_elem)

module_props = get_fem_props()

jive = CJiveRunner(module_props, elems=obs_elems)
globdat = jive()
u_obs = globdat["state0"]
K_obs = globdat["matrix0"]
n_obs = len(u_obs)
alpha2_mle = u_obs @ K_obs @ u_obs / n_obs

plot_nodes, plot_elems = mesh_interval_with_line2(n=720 * n_elem)
plot_jive_runner = CJiveRunner(module_props, elems=plot_elems)
plot_globdat = plot_jive_runner()

for r in [2, 16, 90]:
    ref_nodes, ref_elems = mesh_interval_with_line2(n=r * n_elem)

    module_props = get_fem_props()
    model_props = module_props.pop("model")
    ref_jive_runner = CJiveRunner(module_props, elems=ref_elems)
    obs_jive_runner = CJiveRunner(module_props, elems=obs_elems)

    inf_cov = InverseCovarianceOperator(model_props=model_props, scale=alpha2_mle)
    inf_prior = GaussianProcess(None, inf_cov)
    ref_prior = ProjectedPrior(prior=inf_prior, jive_runner=ref_jive_runner)
    ref_globdat = ref_prior.globdat
    obs_prior = ProjectedPrior(prior=inf_prior, jive_runner=obs_jive_runner)
    obs_globdat = obs_prior.globdat

    H_obs, f_obs = compute_bfem_observations(obs_prior, ref_prior)

    u_coarse = obs_globdat["state0"]
    u = ref_globdat["state0"]
    Phi = create_phi_from_globdat(obs_globdat, ref_globdat)
    u_coarse = Phi @ u_coarse

    posterior = ref_prior.condition_on(H_obs, f_obs)

    Phi_plot = create_phi_from_globdat(ref_globdat, plot_globdat)
    prior = ref_prior @ Phi_plot.T
    posterior = posterior @ Phi_plot.T

    samples_u_prior = prior.calc_samples(20, 0)
    samples_u_post = posterior.calc_samples(20, 0)

    u_prior = prior.calc_mean()
    u_post = posterior.calc_mean()
    std_u_prior = prior.calc_std()
    std_u_post = posterior.calc_std()

    c = sns.color_palette("rocket_r", n_colors=8)[2]
    x_plot = np.linspace(0, 1, len(u_post))

    xmarkers = np.linspace(0.0, 1.0, 6)
    ymarkers = np.linspace(-1.5, 1.5, 7)

    plt.figure()
    plt.plot(x_plot, u_prior, color=c)
    plt.plot(x_plot, samples_u_prior.T, color=c, linewidth=0.5)
    plt.fill_between(
        x_plot, u_prior - 2 * std_u_prior, u_prior + 2 * std_u_prior, color=c, alpha=0.3
    )
    plt.xlabel(r"$x$", fontsize=12)
    plt.ylabel(r"$u$", fontsize=12)
    plt.xticks(xmarkers)
    plt.yticks(ymarkers)
    plt.ylim(ymarkers[[0, -1]])
    fname = os.path.join("img", "bfem-prior_ref-{}.pdf".format(len(ref_elems)))
    plt.savefig(fname=fname, bbox_inches="tight")
    plt.show()

    xmarkers = np.linspace(0.0, 1.0, 6)
    ymarkers = np.linspace(-1.0, 2.0, 7)

    plt.figure()
    plt.plot(x_plot, u_post, color=c)
    plt.plot(x_plot, samples_u_post.T, color=c, linewidth=0.5)
    plt.fill_between(
        x_plot, u_post - 2 * std_u_post, u_post + 2 * std_u_post, color=c, alpha=0.3
    )
    plt.plot(x_plot, u_exact(x_plot), color="k")
    plt.xlabel(r"$x$", fontsize=12)
    plt.ylabel(r"$u$", fontsize=12)
    plt.xticks(xmarkers)
    plt.yticks(ymarkers)
    plt.ylim(ymarkers[[0, -1]])
    fname = os.path.join("img", "bfem-posterior_ref-{}.pdf".format(len(ref_elems)))
    plt.savefig(fname=fname, bbox_inches="tight")
    plt.show()
