import os
import numpy as np
from scipy.sparse import diags_array

from probability import TemperedPosterior
from probability.multivariate import Gaussian, SymbolicCovariance
from probability.process import GaussianProcess, ZeroMeanFunction, SquaredExponential
from probability.sampling import MCMCRunner
from util.linalg import Matrix

from experiments.inverse.frp_damage.likelihoods import (
    BFEMLikelihoodHierarchical,
    BFEMLikelihoodHeterarchical,
)
from experiments.inverse.frp_damage import caching

n_burn = 10000
n_sample = 20000
std_pd = 1e-6

h_obs = 0.050
sigma_e = 1e-3
seed = 0

hierarchical = False

for seed in range(1):
    if hierarchical:
        h_ref = "{:.3f}r1".format(h_obs)
    else:
        h_ref = "{:.3f}d1".format(h_obs)
        h_hyp = "{:.3f}h1".format(h_obs)

    obs_nodes, obs_elems, obs_egroups = caching.get_or_calc_mesh(h=h_obs)
    egroup_obs = obs_egroups["matrix"]
    obs_distances = caching.get_or_calc_distances(egroup=egroup_obs, h=h_obs)

    if hierarchical:
        ref_nodes, ref_elems, ref_egroups = caching.get_or_calc_mesh(h=h_ref)
        egroup_ref = ref_egroups["matrix"]
        ref_distances = caching.get_or_calc_distances(egroup=egroup_ref, h=h_ref)
    else:
        ref_nodes, ref_elems, ref_egroups = caching.get_or_calc_dual_mesh(h=h_ref)
        egroup_ref = ref_egroups["matrix"]
        ref_distances = caching.get_or_calc_distances(egroup=egroup_ref, h=h_ref)

        hyp_mesh = caching.get_or_calc_hyper_mesh(h=h_hyp, do_groups=True)
        hyp_nodes, hyp_elems, hyp_egroups = hyp_mesh
        egroup_hyp = hyp_egroups["matrix"]
        hyp_distances = caching.get_or_calc_distances(egroup=egroup_hyp, h=h_hyp)

    domain = np.linspace(0.0, 0.2, 101)

    inf_prior = GaussianProcess(
        mean=ZeroMeanFunction(),
        cov=SquaredExponential(l=0.02, sigma=2.0),
    )

    U, s, _ = np.linalg.svd(inf_prior.calc_cov(domain, domain))

    trunc = 10
    eigenfuncs = U[:, :trunc]
    eigenvalues = s[:trunc]

    kl_cov = SymbolicCovariance(Matrix(diags_array(eigenvalues), name="S"))
    kl_prior = Gaussian(mean=None, cov=kl_cov)

    #########################
    # get precomputed stuff #
    #########################

    obs_ipoints = caching.get_or_calc_ipoints(egroup=egroup_obs, h=h_obs)
    obs_distances = caching.get_or_calc_distances(egroup=egroup_obs, h=h_obs)

    ref_ipoints = caching.get_or_calc_ipoints(egroup=egroup_ref, h=h_ref)
    ref_distances = caching.get_or_calc_distances(egroup=egroup_ref, h=h_ref)

    if not hierarchical:
        hyp_ipoints = caching.get_or_calc_ipoints(egroup=egroup_hyp, h=h_hyp)
        hyp_distances = caching.get_or_calc_distances(egroup=egroup_hyp, h=h_hyp)

    obs_backdoor = {}
    obs_backdoor["xcoord"] = obs_ipoints[:, 0]
    obs_backdoor["ycoord"] = obs_ipoints[:, 1]
    obs_backdoor["e"] = np.zeros(obs_ipoints.shape[0])

    ref_backdoor = {}
    ref_backdoor["xcoord"] = ref_ipoints[:, 0]
    ref_backdoor["ycoord"] = ref_ipoints[:, 1]
    ref_backdoor["e"] = np.zeros(ref_ipoints.shape[0])

    if not hierarchical:
        hyp_backdoor = {}
        hyp_backdoor["xcoord"] = hyp_ipoints[:, 0]
        hyp_backdoor["ycoord"] = hyp_ipoints[:, 1]
        hyp_backdoor["e"] = np.zeros(hyp_ipoints.shape[0])

    if hierarchical:
        operator = caching.get_or_calc_dic_operator(elems=ref_elems, h=h_ref)
    else:
        operator = caching.get_or_calc_dic_operator(elems=hyp_elems, h=h_hyp)

    truth = caching.get_or_calc_true_dic_observations(h=0.002)

    if hierarchical:
        likelihood = BFEMLikelihoodHierarchical(
            operator=operator,
            observations=truth,
            sigma_e=sigma_e,
            obs_ipoints=obs_ipoints,
            ref_ipoints=ref_ipoints,
            obs_distances=obs_distances,
            ref_distances=ref_distances,
            eigenfuncs=eigenfuncs,
            domain=domain,
            obs_egroups=obs_egroups,
            ref_egroups=ref_egroups,
            obs_backdoor=obs_backdoor,
            ref_backdoor=ref_backdoor,
        )
    else:
        likelihood = BFEMLikelihoodHeterarchical(
            operator=operator,
            observations=truth,
            sigma_e=sigma_e,
            obs_ipoints=obs_ipoints,
            ref_ipoints=ref_ipoints,
            hyp_ipoints=hyp_ipoints,
            obs_distances=obs_distances,
            ref_distances=ref_distances,
            hyp_distances=hyp_distances,
            eigenfuncs=eigenfuncs,
            domain=domain,
            obs_egroups=obs_egroups,
            ref_egroups=ref_egroups,
            hyp_egroups=hyp_egroups,
            obs_backdoor=obs_backdoor,
            ref_backdoor=ref_backdoor,
            hyp_backdoor=hyp_backdoor,
        )

    def linear_tempering(i):
        if i < n_burn:
            return i / n_burn
        else:
            return 1.0

    def stepwise_tempering(i, *, n_step=100):
        # divide the tempering period into 100 steps
        if i < n_burn:
            return ((i * n_step) // n_burn) / n_step
        else:
            return 1.0

    rng = np.random.default_rng(seed)
    start_value = kl_prior.calc_sample(rng)
    target = TemperedPosterior(kl_prior, likelihood)
    proposal = Gaussian(None, kl_prior.calc_cov().toarray())

    if hierarchical:
        fname = "checkpoint_bfem-hier_h-{:.3f}_noise-{:.0e}_seed-{}.pkl"
    else:
        fname = "checkpoint_bfem-heter_h-{:.3f}_noise-{:.0e}_seed-{}.pkl"
    fname = os.path.join("checkpoints", fname.format(h_obs, sigma_e, seed))

    mcmc = MCMCRunner(
        target=target,
        proposal=proposal,
        n_sample=n_sample,
        n_burn=n_burn,
        start_value=start_value,
        seed=rng,
        tempering=stepwise_tempering,
        return_info=True,
        checkpoint=fname,
    )

    samples, info = mcmc()

    if hierarchical:
        fname = "posterior-samples_bfem-hier_h-{:.3f}_noise-{:.0e}_seed-{}.npy"
    else:
        fname = "posterior-samples_bfem-heter_h-{:.3f}_noise-{:.0e}_seed-{}.npy"
    fname = os.path.join("output", fname.format(h_obs, sigma_e, seed))
    np.save(fname, samples)

    if hierarchical:
        fname = "posterior-logpdfs_bfem-hier_h-{:.3f}_noise-{:.0e}_seed-{}.npy"
    else:
        fname = "posterior-logpdfs_bfem-heter_h-{:.3f}_noise-{:.0e}_seed-{}.npy"
    fname = os.path.join("output", fname.format(h_obs, sigma_e, seed))
    np.save(fname, info["loglikelihood"])
