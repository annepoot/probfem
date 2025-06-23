import os
import numpy as np
from scipy.sparse import diags_array

from probability import TemperedPosterior
from probability.multivariate import Gaussian, SymbolicCovariance
from probability.process import GaussianProcess, ZeroMeanFunction, SquaredExponential
from probability.sampling import MCMCRunner
from util.linalg import Matrix

from experiments.inverse.frp_damage import caching
from experiments.inverse.frp_damage.likelihoods import BPODLikelihoodHierarchical

n_burn = 10000
n_sample = 20000
std_pd = 1e-6

h = 0.01
k = 10
l = 10
sigma_e = 1e-3
seed = 0

for seed in range(10):
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

    likelihood = BPODLikelihoodHierarchical(
        operator=obs_operator,
        observations=truth,
        sigma_e=sigma_e,
        basis=basis,
        k=k,
        l=l,
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

    fname = "checkpoint_bpod_h-{:.3f}_noise-{:.0e}_k-{}_l-{}_seed-{}.pkl"
    fname = os.path.join("checkpoints", fname.format(h, sigma_e, k, l, seed))

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

    fname = "posterior-samples_bpod_h-{:.3f}_noise-{:.0e}_k-{}_l-{}_seed-{}.npy"
    fname = os.path.join("output", fname.format(h, sigma_e, k, l, seed))
    np.save(fname, samples)

    fname = "posterior-logpdfs_bpod_h-{:.3f}_noise-{:.0e}_k-{}_l-{}_seed-{}.npy"
    fname = os.path.join("output", fname.format(h, sigma_e, k, l, seed))
    np.save(fname, info["loglikelihood"])
