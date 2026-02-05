import sys
import os
import numpy as np
from scipy.sparse import diags_array
import itertools

from probability import TemperedPosterior
from probability.multivariate import Gaussian, SymbolicCovariance, Mixture
from probability.process import GaussianProcess, ZeroMeanFunction, SquaredExponential
from probability.sampling import RandomWalkMetropolisSampler, MetropolisHastingsSampler
from util.linalg import Matrix

from experiments.reproduction.nonhierarchical.frp_dic import caching
from experiments.reproduction.nonhierarchical.frp_dic.likelihoods import FEMLikelihood

n_burn = 10000
n_sample = 20000
std_pd = 1e-6

hs = [0.100, 0.050, 0.020, 0.010]
sigma_es = [1e-2, 1e-3, 1e-4]
# seeds = range(10)  # single run
seeds = ["0-10"]  # single run

combis = list(itertools.product(hs, sigma_es, seeds))

if __name__ == "__main__":
    run_idx = int(sys.argv[1])
    job_id = int(sys.argv[2])
    h, sigma_e, seed = combis[run_idx]

    print("############")
    print("# SETTINGS #")
    print("############")
    print("run idx:\t", run_idx)
    print("job id: \t", job_id)
    print("h:      \t", h)
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

    backdoor = {}
    backdoor["xcoord"] = ipoints[:, 0]
    backdoor["ycoord"] = ipoints[:, 1]
    backdoor["e"] = np.zeros(ipoints.shape[0])

    obs_operator = caching.get_or_calc_dic_operator(elems=elems, h=h)
    truth = caching.get_or_calc_true_dic_observations(h=0.002)

    likelihood = FEMLikelihood(
        operator=obs_operator,
        observations=truth,
        sigma_e=sigma_e,
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

    target = TemperedPosterior(kl_prior, likelihood)

    fname_cp = "checkpoint_fem_h-{:.3f}_noise-{:.0e}_seed-{}.pkl"
    fname_cp = os.path.join("checkpoints", fname_cp.format(h, sigma_e, seed))

    if isinstance(seed, int):
        rng = np.random.default_rng(seed)
        start_value = kl_prior.calc_sample(rng)
        proposal = Gaussian(None, kl_prior.calc_cov().toarray())

        mcmc = RandomWalkMetropolisSampler(
            target=target,
            proposal=proposal,
            n_sample=n_sample,
            n_burn=n_burn,
            start_value=start_value,
            seed=rng,
            tempering=linear_tempering,
            return_info=True,
            checkpoint=fname_cp,
        )

    elif isinstance(seed, str):
        start, end = [int(i) for i in seed.split("-")]
        ensemble = []
        sample_list = []
        cov_list = []

        for i in range(start, end):
            fname = "posterior-samples_{}_h-{:.3f}_noise-{:.0e}_seed-{}.npy"
            fname = fname.format("fem", h, sigma_e, i)
            fname = os.path.join("output", "fem", fname)

            samples = np.load(fname)
            samples = samples[n_burn:]

            sample_list.append(samples)

            sample_mean = np.mean(samples, axis=0)
            sample_cov = np.cov(samples.T)

            ensemble.append(Gaussian(sample_mean, sample_cov))
            cov_list.append(sample_cov)

        proposal_ind = Mixture(ensemble)
        proposal_rw = Gaussian(None, np.mean(cov_list, axis=0))

        # halve burn-in phase
        rng = np.random.default_rng(end)
        start_value = proposal_ind.calc_sample(rng)

        mcmc = MetropolisHastingsSampler(
            target=target,
            proposal_rw=proposal_rw,
            proposal_ind=proposal_ind,
            beta=0.5,
            n_sample=n_sample // 2,
            n_burn=n_burn // 2,
            start_value=start_value,
            seed=rng,
            tempering=linear_tempering,
            return_info=True,
            checkpoint=fname_cp,
        )
    else:
        assert False

    samples, info = mcmc()

    outdir = os.path.join("output", str(job_id))
    os.makedirs(outdir, exist_ok=True)

    fname = "posterior-samples_fem_h-{:.3f}_noise-{:.0e}_seed-{}.npy"
    fname = os.path.join(outdir, fname.format(h, sigma_e, seed))
    np.save(fname, samples)

    fname = "posterior-logpdfs_fem_h-{:.3f}_noise-{:.0e}_seed-{}.npy"
    fname = os.path.join(outdir, fname.format(h, sigma_e, seed))
    np.save(fname, info["loglikelihood"])
