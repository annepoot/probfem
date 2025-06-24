import numpy as np

from bfem import compute_bfem_observations
from fem.jive import CJiveRunner
from probability.process import (
    InverseCovarianceOperator,
    GaussianProcess,
    ProjectedPrior,
)
from experiments.inverse.frp_damage.props import get_fem_props
from experiments.inverse.frp_damage import caching, misc, params

props = get_fem_props()

n_fiber = params.geometry_params["n_fiber"]
r_fiber = params.geometry_params["r_fiber"]
h_obs = 0.05
h_ref = "{:.3f}r1".format(h_obs)

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

ref_nodes, ref_elems, ref_egroups = caching.get_or_calc_mesh(h=h_ref)
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

obs_module_props = get_fem_props()
ref_module_props = get_fem_props()

obs_model_props = obs_module_props.pop("model")
ref_model_props = ref_module_props.pop("model")

assert obs_model_props == ref_model_props

obs_jive_runner = CJiveRunner(obs_module_props, elems=obs_elems, egroups=obs_egroups)
ref_jive_runner = CJiveRunner(ref_module_props, elems=ref_elems, egroups=ref_egroups)

inf_cov = InverseCovarianceOperator(model_props=ref_model_props, scale=1.0)
inf_prior = GaussianProcess(None, inf_cov)

obs_prior = ProjectedPrior(prior=inf_prior, jive_runner=obs_jive_runner, **obs_backdoor)
ref_prior = ProjectedPrior(prior=inf_prior, jive_runner=ref_jive_runner, **ref_backdoor)

obsdat = obs_prior.globdat
refdat = ref_prior.globdat

u_obs = obsdat["state0"]
K_obs = obsdat["matrix0"]
n_obs = len(u_obs)
alpha2_mle = u_obs @ K_obs @ u_obs / n_obs

assert ref_prior.prior.cov.scale == 1.0
ref_prior.prior.cov.scale = alpha2_mle
assert obs_prior.prior.cov.scale == alpha2_mle

obs_prior.recompute_moments(**obs_backdoor)
ref_prior.recompute_moments(**ref_backdoor)

H_obs, f_obs = compute_bfem_observations(obs_prior, ref_prior)
posterior = ref_prior.condition_on(H_obs, f_obs)

samples = posterior.calc_samples(n=1000, seed=0)

mean = np.mean(samples, axis=0)
std = np.std(samples, axis=0)

for i in range(20):
    QuickViewer(
        samples[i],
        ref_globdat,
        comp=0,
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
)
