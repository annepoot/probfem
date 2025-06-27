import sys
import os
import numpy as np
from scipy.sparse import diags_array
import itertools

from probability import TemperedPosterior
from probability.multivariate import Gaussian, SymbolicCovariance
from probability.process import GaussianProcess, ZeroMeanFunction, SquaredExponential
from probability.sampling import MCMCRunner
from util.linalg import Matrix

from experiments.inverse.frp_damage import caching
from experiments.inverse.frp_damage.likelihoods import BFEMLikelihoodHierarchical

n_burn = 10000
n_sample = 20000
std_pd = 1e-6

sigma_e = 1e-3

hs = [0.100, 0.050, 0.020]
seeds = range(10)

combis = list(itertools.product(hs, seeds))

if __name__ == "__main__":
    run_idx = int(sys.argv[1])
    job_id = int(sys.argv[2])
    h_obs, seed = combis[run_idx]

    print("############")
    print("# SETTINGS #")
    print("############")
    print("run idx:\t", run_idx)
    print("job id: \t", job_id)
    print("h_obs:  \t", h_obs)
    print("seed:   \t", seed)
    print("")

    h_ref = "{:.3f}r1".format(h_obs)

    obs_nodes, obs_elems, obs_egroups = caching.get_or_calc_mesh(h=h_obs)
    egroup_obs = obs_egroups["matrix"]
    obs_distances = caching.get_or_calc_distances(egroup=egroup_obs, h=h_obs)

    ref_nodes, ref_elems, ref_egroups = caching.get_or_calc_mesh(h=h_ref)
    egroup_ref = ref_egroups["matrix"]
    ref_distances = caching.get_or_calc_distances(egroup=egroup_ref, h=h_ref)

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

    obs_backdoor = {}
    obs_backdoor["xcoord"] = obs_ipoints[:, 0]
    obs_backdoor["ycoord"] = obs_ipoints[:, 1]
    obs_backdoor["e"] = np.zeros(obs_ipoints.shape[0])

    ref_backdoor = {}
    ref_backdoor["xcoord"] = ref_ipoints[:, 0]
    ref_backdoor["ycoord"] = ref_ipoints[:, 1]
    ref_backdoor["e"] = np.zeros(ref_ipoints.shape[0])

    obs_operator = caching.get_or_calc_dic_operator(elems=obs_elems, h=h_obs)
    ref_operator = caching.get_or_calc_dic_operator(elems=ref_elems, h=h_ref)

    truth = caching.get_or_calc_true_dic_observations(h=0.002)

    likelihood = BFEMLikelihoodHierarchical(
        obs_operator=obs_operator,
        ref_operator=ref_operator,
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

    fname = "checkpoint_bfem_h-{:.3f}_noise-{:.0e}_seed-{}.pkl"
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

    outdir = os.path.join("output", str(job_id))
    os.makedirs(outdir, exist_ok=True)

    fname = "posterior-samples_bfem_h-{:.3f}_noise-{:.0e}_seed-{}.npy"
    fname = os.path.join(outdir, fname.format(h_obs, sigma_e, seed))
    np.save(fname, samples)

    fname = "posterior-logpdfs_bfem_h-{:.3f}_noise-{:.0e}_seed-{}.npy"
    fname = os.path.join(outdir, fname.format(h_obs, sigma_e, seed))
    np.save(fname, info["loglikelihood"])
