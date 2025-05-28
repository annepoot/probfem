import sys
import os
import numpy as np
from scipy.sparse import eye_array, diags_array
import itertools

from fem.jive import CJiveRunner
from probability import Likelihood, TemperedPosterior
from probability.multivariate import Gaussian, SymbolicCovariance
from probability.process import GaussianProcess, ZeroMeanFunction, SquaredExponential
from probability.sampling import MCMCRunner
from util.linalg import Matrix

from experiments.inverse.frp_damage.props import get_fem_props
from experiments.inverse.frp_damage import caching, misc, params

n_burn = 10000
n_sample = 20000
std_pd = 1e-6

hs = [0.050, 0.020, 0.010]
sigma_es = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
seeds = range(10)

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

    class CustomLikelihood(Likelihood):

        def __init__(self):
            self.ipoints = ipoints
            self.distances = distances
            self.operator = obs_operator
            self.observations = truth
            n_obs = len(self.observations)
            self.noise = SymbolicCovariance(Matrix(sigma_e**2 * eye_array(n_obs)))
            self.dist = Gaussian(self.observations, self.noise)
            self.eigenfuncs = eigenfuncs

            self._props = get_fem_props()
            self._E_matrix = params.material_params["E_matrix"]
            self._damage_map = misc.calc_damage_map(ipoints, distances, domain)

        def calc_logpdf(self, x):
            damage = misc.sigmoid(self.eigenfuncs @ x, 1.0, 0.0)
            backdoor["e"] = self._E_matrix * (1 - self._damage_map @ damage)

            jive = CJiveRunner(self._props, elems=elems, egroups=egroups)
            globdat = jive(**backdoor)

            state0 = globdat["state0"]
            pred = self.operator @ state0

            loglikelihood = self.dist.calc_logpdf(pred)
            return loglikelihood

    likelihood = CustomLikelihood()

    def linear_tempering(i):
        if i < n_burn:
            return i / n_burn
        else:
            return 1.0

    rng = np.random.default_rng(seed)
    start_value = kl_prior.calc_sample(rng)
    target = TemperedPosterior(kl_prior, likelihood)
    proposal = Gaussian(None, kl_prior.calc_cov().toarray())

    mcmc = MCMCRunner(
        target=target,
        proposal=proposal,
        n_sample=n_sample,
        n_burn=n_burn,
        start_value=start_value,
        seed=rng,
        tempering=linear_tempering,
        return_info=True,
    )

    samples, info = mcmc()

    outdir = os.path.join("output", str(job_id))
    os.makedirs(outdir, exist_ok=True)

    fname = "posterior-samples_fem_h-{:.3f}_noise-{:.0e}_seed-{}.npy"
    fname = os.path.join(outdir, fname.format(h, sigma_e, seed))
    np.save(fname, samples)

    fname = "posterior-logpdfs_fem_h-{:.3f}_noise-{:.0e}_seed-{}.npy"
    fname = os.path.join(outdir, fname.format(h, sigma_e, seed))
    np.save(fname, info["loglikelihood"])
