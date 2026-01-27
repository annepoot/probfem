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
from experiments.inverse.frp_damage.likelihoods import PODLikelihood

n_burn = 10000
n_sample = 20000
std_pd = 1e-6

h = 0.010
sigma_e = 1e-4

ks = [1, 2, 5, 10, 20, 50, 100]
seeds = range(10)

combis = list(itertools.product(ks, seeds))

if __name__ == "__main__":
    run_idx = int(sys.argv[1])
    job_id = int(sys.argv[2])
    k, seed = combis[run_idx]

    print("############")
    print("# SETTINGS #")
    print("############")
    print("run idx:\t", run_idx)
    print("job id: \t", job_id)
    print("k:      \t", k)
    print("sigma_e:\t", sigma_e)
    print("seed:   \t", seed)
    print("")

    nodes, elems, egroups = caching.get_or_calc_mesh(h=h)
    egroup = egroups["matrix"]
    distances = caching.get_or_calc_distances(egroup=egroup, h=h)

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

    ipoints = caching.get_or_calc_ipoints(egroup=egroup, h=h)
    distances = caching.get_or_calc_distances(egroup=egroup, h=h)
    basis = caching.get_or_calc_pod_basis(h=h)
    lifting = caching.get_or_calc_pod_lifting(h=h)

    backdoor = {}
    backdoor["xcoord"] = ipoints[:, 0]
    backdoor["ycoord"] = ipoints[:, 1]
    backdoor["e"] = np.zeros(ipoints.shape[0])

    obs_operator = caching.get_or_calc_dic_operator(elems=elems, h=h)
    truth = caching.get_or_calc_true_dic_observations(h=0.002)

    likelihood = PODLikelihood(
        operator=obs_operator,
        observations=truth,
        sigma_e=sigma_e,
        basis=basis,
        k=k,
        lifting=lifting,
        ipoints=ipoints,
        distances=distances,
        eigenfuncs=eigenfuncs,
        domain=domain,
        egroups=egroups,
        backdoor=backdoor,
    )

    def linear_tempering(i):
        if i < n_burn:
            return i / n_burn
        else:
            return 1.0

    rng = np.random.default_rng(seed)
    start_value = kl_prior.calc_sample(rng)
    target = TemperedPosterior(kl_prior, likelihood)
    proposal = Gaussian(None, kl_prior.calc_cov().toarray())

    fname = "checkpoint_pod_h-{:.3f}_noise-{:.0e}_k-{}_seed-{}.pkl"
    fname = os.path.join("checkpoints", fname.format(h, sigma_e, k, seed))

    mcmc = MCMCRunner(
        target=target,
        proposal=proposal,
        n_sample=n_sample,
        n_burn=n_burn,
        start_value=start_value,
        seed=rng,
        tempering=linear_tempering,
        return_info=True,
        checkpoint=fname,
    )

    samples, info = mcmc()

    outdir = os.path.join("output", str(job_id))
    os.makedirs(outdir, exist_ok=True)

    fname = "posterior-samples_pod_h-{:.3f}_noise-{:.0e}_k-{}_seed-{}.npy"
    fname = os.path.join(outdir, fname.format(h, sigma_e, k, seed))
    np.save(fname, samples)

    fname = "posterior-logpdfs_pod_h-{:.3f}_noise-{:.0e}_k-{}_seed-{}.npy"
    fname = os.path.join(outdir, fname.format(h, sigma_e, k, seed))
    np.save(fname, info["loglikelihood"])
