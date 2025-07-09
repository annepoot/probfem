import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from bfem.observation import compute_bfem_observations
from experiments.reproduction.nonhierarchical.pullout_bar.props import get_fem_props
from fem.jive import CJiveRunner
from fem.meshing import (
    mesh_interval_with_line2,
    create_phi_from_globdat,
    create_hypermesh,
)
from probability.multivariate import Gaussian
from probability.process import (
    GaussianProcess,
    InverseCovarianceOperator,
    ProjectedPrior,
)
from util.linalg import Matrix

from experiments.reproduction.nonhierarchical.pullout_bar import misc

props = get_fem_props()
k = props["model"]["model"]["spring"]["k"]
E = props["model"]["model"]["elastic"]["material"]["E"]
f = props["model"]["model"]["neum"]["initLoad"]

n_elem = 4

# options: exact, hierarchical, inverted, random
ref_type = "exact"

obs_nodes, obs_elems = mesh_interval_with_line2(n=n_elem)

if ref_type == "exact":
    ref_nodes, ref_elems = mesh_interval_with_line2(n=720)
elif ref_type == "hierarchical":
    ref_nodes, ref_elems = mesh_interval_with_line2(n=2 * n_elem)
elif ref_type == "inverted":
    ref_nodes, ref_elems = misc.invert_mesh(obs_elems)
elif ref_type == "random":
    ref_nodes, ref_elems = misc.random_mesh(n=n_elem, seed=0)
else:
    assert False

(hyp_nodes, hyp_elems), hyp_map = create_hypermesh(obs_elems, ref_elems)

module_props = get_fem_props()

jive = CJiveRunner(module_props, elems=obs_elems)
globdat = jive()
u_obs = globdat["state0"]
K_obs = globdat["matrix0"]
n_obs = len(u_obs)
alpha2_mle = u_obs @ K_obs @ u_obs / n_obs

plot_nodes, plot_elems = mesh_interval_with_line2(n=720 * n_elem)

module_props = get_fem_props()
plot_jive_runner = CJiveRunner(module_props, elems=plot_elems)
plot_globdat = plot_jive_runner()

model_props = module_props.pop("model")
ref_jive_runner = CJiveRunner(module_props, elems=ref_elems)
obs_jive_runner = CJiveRunner(module_props, elems=obs_elems)
hyp_jive_runner = CJiveRunner(module_props, elems=hyp_elems)

inf_cov = InverseCovarianceOperator(model_props=model_props, scale=alpha2_mle)
inf_prior = GaussianProcess(None, inf_cov)
obs_prior = ProjectedPrior(prior=inf_prior, jive_runner=obs_jive_runner)
obs_globdat = obs_prior.globdat
ref_prior = ProjectedPrior(prior=inf_prior, jive_runner=ref_jive_runner)
ref_globdat = ref_prior.globdat
hyp_prior = ProjectedPrior(prior=inf_prior, jive_runner=hyp_jive_runner)
hyp_globdat = hyp_prior.globdat

H_obs, f_obs = compute_bfem_observations(obs_prior, hyp_prior)
H_ref, f_ref = compute_bfem_observations(ref_prior, hyp_prior)

Phi_obs = H_obs[0].T
Phi_ref = H_ref[0].T
K_hyp = H_obs[1]

K_obs = Matrix((Phi_obs.T @ K_hyp @ Phi_obs).evaluate(), name="K_obs")
K_ref = Matrix((Phi_ref.T @ K_hyp @ Phi_ref).evaluate(), name="K_ref")
K_x = Matrix((Phi_ref.T @ K_hyp @ Phi_obs).evaluate(), name="K_x")

P_obs = Phi_obs @ K_obs.inv @ Phi_obs.T @ K_hyp
P_ref = Phi_ref @ K_ref.inv @ Phi_ref.T @ K_hyp

mean = Phi_obs @ obs_globdat["state0"]
cov = K_ref.inv.evaluate()
cov -= (K_ref.inv @ K_x @ K_obs.inv @ K_x.T @ K_ref.inv).evaluate()
cov *= alpha2_mle
cov = Phi_ref @ (Phi_ref @ cov).T

Phi_plot = create_phi_from_globdat(hyp_globdat, plot_globdat)

posterior = Gaussian(mean, cov, allow_singular=True) @ Phi_plot.T

samples_u_post = posterior.calc_samples(20, 0)
u_post = posterior.calc_mean()
std_u_post = posterior.calc_std()

c = sns.color_palette("rocket_r", n_colors=8)[2]
x_plot = np.linspace(0, 1, len(u_post))
u_exact = misc.u_exact(x_plot, k=k, E=E, f=f)

xmarkers = np.linspace(0.0, 1.0, 6)
ymarkers = np.linspace(-1.0, 2.0, 7)

plt.figure()
plt.plot(x_plot, u_post, color=c)
plt.plot(x_plot, samples_u_post.T, color=c, linewidth=0.5)
plt.fill_between(
    x_plot, u_post - 2 * std_u_post, u_post + 2 * std_u_post, color=c, alpha=0.3
)
plt.plot(x_plot, u_exact, color="k")
plt.xlabel(r"$x$", fontsize=12)
plt.ylabel(r"$u$", fontsize=12)
plt.xticks(xmarkers)
plt.yticks(ymarkers)
plt.ylim(ymarkers[[0, -1]])
fname = "bfem-posterior_{}.pdf".format(ref_type)
fname = os.path.join("img", fname)
plt.savefig(fname=fname, bbox_inches="tight")
plt.show()
