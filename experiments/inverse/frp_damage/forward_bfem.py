import numpy as np
from scipy.stats import multivariate_normal

from bfem import compute_bfem_observations
from fem.jive import CJiveRunner
from probability.process import (
    InverseCovarianceOperator,
    GaussianProcess,
    ProjectedPrior,
)
from experiments.inverse.frp_damage.props import get_fem_props
from experiments.inverse.frp_damage import caching, misc, params
from util.linalg import Matrix

# options: refined, shifted, flipped
ref_type = "flipped"

props = get_fem_props()

n_fiber = params.geometry_params["n_fiber"]
r_fiber = params.geometry_params["r_fiber"]
h_obs = 0.050

obs_nodes, obs_elems, obs_egroups = caching.get_or_calc_mesh(h=h_obs)
obs_egroup = obs_egroups["matrix"]

obs_ipoints = caching.get_or_calc_ipoints(egroup=obs_egroup, h=h_obs)
obs_ip_stiffnesses = caching.get_or_calc_true_stiffnesses(egroup=obs_egroup, h=h_obs)

obs_backdoor = {}
obs_backdoor["xcoord"] = obs_ipoints[:, 0]
obs_backdoor["ycoord"] = obs_ipoints[:, 1]
obs_backdoor["e"] = obs_ip_stiffnesses

props = get_fem_props()
obs_jive = CJiveRunner(props, elems=obs_elems, egroups=obs_egroups)
obs_globdat = obs_jive(**obs_backdoor)

if ref_type == "refined":
    h_ref = "{:.3f}r1".format(h_obs)
    ref_nodes, ref_elems, ref_egroups = caching.get_or_calc_mesh(h=h_ref)
elif ref_type == "shifted":
    h_ref = "{:.3f}d1".format(h_obs)
    ref_nodes, ref_elems, ref_egroups = caching.get_or_calc_dual_mesh(h=h_ref)
elif ref_type == "flipped":
    h_ref = "{:.3f}d2".format(h_obs)
    ref_nodes, ref_elems, ref_egroups = caching.get_or_calc_dual_mesh(h=h_ref)
else:
    assert False

ref_egroup = ref_egroups["matrix"]
ref_ipoints = caching.get_or_calc_ipoints(egroup=ref_egroup, h=h_ref)
ref_ip_stiffnesses = caching.get_or_calc_true_stiffnesses(egroup=ref_egroup, h=h_ref)

ref_backdoor = {}
ref_backdoor["xcoord"] = ref_ipoints[:, 0]
ref_backdoor["ycoord"] = ref_ipoints[:, 1]
ref_backdoor["e"] = ref_ip_stiffnesses

props = get_fem_props()
ref_jive = CJiveRunner(props, elems=ref_elems, egroups=ref_egroups)
ref_globdat = ref_jive(**ref_backdoor)

ref_elem_stiffnesses = misc.calc_elem_stiffnesses(ref_ip_stiffnesses, ref_egroups)

from myjivex.util import QuickViewer, ElemViewer

QuickViewer(
    obs_globdat["state0"],
    obs_globdat,
    comp=0,
)
QuickViewer(
    ref_globdat["state0"],
    ref_globdat,
    comp=0,
)
ElemViewer(
    ref_elem_stiffnesses,
    ref_globdat,
    maxcolor=params.material_params["E_matrix"],
    title=r"stiffness, $N_e = {}$".format(len(ref_elems)),
)

if ref_type == "refined":
    h_hyp = "{:.3f}r1".format(h_obs)
    hyp_mesh = ref_nodes, ref_elems, ref_egroups
elif ref_type == "shifted":
    h_hyp = "{:.3f}h1".format(h_obs)
    hyp_mesh = caching.get_or_calc_hyper_mesh(h=h_hyp, do_groups=True)
elif ref_type == "flipped":
    h_hyp = "{:.3f}h2".format(h_obs)
    hyp_mesh = caching.get_or_calc_hyper_mesh(h=h_hyp, do_groups=True)
else:
    assert False

hyp_nodes, hyp_elems, hyp_egroups = hyp_mesh
hyp_egroup = hyp_egroups["matrix"]

hyp_ipoints = caching.get_or_calc_ipoints(egroup=hyp_egroup, h=h_hyp)
hyp_ip_stiffnesses = caching.get_or_calc_true_stiffnesses(egroup=hyp_egroup, h=h_hyp)

hyp_backdoor = {}
hyp_backdoor["xcoord"] = hyp_ipoints[:, 0]
hyp_backdoor["ycoord"] = hyp_ipoints[:, 1]
hyp_backdoor["e"] = hyp_ip_stiffnesses

props = get_fem_props()
props["usermodules"]["solver"]["solver"] = {
    "type": "GMRES",
    "precision": 1e100,
}
hyp_jive = CJiveRunner(props, elems=hyp_elems, egroups=hyp_egroups)
hyp_globdat = hyp_jive(**hyp_backdoor)

hyp_elem_stiffnesses = misc.calc_elem_stiffnesses(hyp_ip_stiffnesses, hyp_egroups)

obs_module_props = get_fem_props()
ref_module_props = get_fem_props()
hyp_module_props = get_fem_props()
hyp_module_props["usermodules"]["solver"]["solver"] = {
    "type": "GMRES",
    "precision": 1e100,
}

obs_model_props = obs_module_props.pop("model")
ref_model_props = ref_module_props.pop("model")
hyp_model_props = hyp_module_props.pop("model")

assert obs_model_props == ref_model_props == hyp_model_props

obs_jive_runner = CJiveRunner(obs_module_props, elems=obs_elems, egroups=obs_egroups)
ref_jive_runner = CJiveRunner(ref_module_props, elems=ref_elems, egroups=ref_egroups)
hyp_jive_runner = CJiveRunner(hyp_module_props, elems=hyp_elems, egroups=hyp_egroups)

inf_cov = InverseCovarianceOperator(model_props=ref_model_props, scale=1.0)
inf_prior = GaussianProcess(None, inf_cov)

obs_prior = ProjectedPrior(prior=inf_prior, jive_runner=obs_jive_runner, **obs_backdoor)
ref_prior = ProjectedPrior(prior=inf_prior, jive_runner=ref_jive_runner, **ref_backdoor)
hyp_prior = ProjectedPrior(prior=inf_prior, jive_runner=hyp_jive_runner, **hyp_backdoor)

obsdat = obs_prior.globdat
refdat = ref_prior.globdat
hypdat = hyp_prior.globdat

u_obs = obsdat["state0"]
K_obs = obsdat["matrix0"]
n_obs = len(u_obs)
alpha2_mle = u_obs @ K_obs @ u_obs / n_obs

assert ref_prior.prior.cov.scale == 1.0
ref_prior.prior.cov.scale = alpha2_mle
assert obs_prior.prior.cov.scale == alpha2_mle

obs_prior.recompute_moments(**obs_backdoor)
ref_prior.recompute_moments(**ref_backdoor)
hyp_prior.recompute_moments(**hyp_backdoor)

if ref_type == "refined":
    H_obs, f_obs = compute_bfem_observations(obs_prior, ref_prior)
    posterior = ref_prior.condition_on(H_obs, f_obs)

    samples = posterior.calc_samples(n=1000, seed=0)

    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)

    for i in range(5):
        QuickViewer(
            samples[i] - mean,
            ref_globdat,
            comp=0,
            fname="img/bfem-hier-posterior_sample-{}_h-{:.3f}.png".format(i, h_obs),
        )

    QuickViewer(
        mean,
        ref_globdat,
        comp=0,
    )

    QuickViewer(
        std,
        ref_globdat,
        comp=0,
        maxcolor=2e-4,
        fname="img/bfem-hier-posterior_std_h-{:.3f}.png".format(h_obs),
    )
else:
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

    cov = K_ref.inv.evaluate()
    cov -= (K_ref.inv @ K_x @ K_obs.inv @ K_x.T @ K_ref.inv).evaluate()
    cov *= alpha2_mle

    mean = obs_globdat["state0"]

    mvn = multivariate_normal(None, cov, allow_singular=True)
    samples = mvn.rvs(100, random_state=0)
    samples = (Phi_ref @ samples.T).T

    for i in range(5):
        QuickViewer(
            samples[i],
            hyp_globdat,
            comp=0,
            fname="img/bfem-heter-posterior_sample-{}_h-{:.3f}.png".format(i, h_obs),
        )

    QuickViewer(
        np.std(samples, axis=0),
        hyp_globdat,
        maxcolor=2e-4,
        comp=0,
        fname="img/bfem-heter-posterior_std_h-{:.3f}.png".format(h_obs),
    )
