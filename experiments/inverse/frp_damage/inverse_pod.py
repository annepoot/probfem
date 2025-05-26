import os
import numpy as np
from scipy.sparse import eye_array, diags_array

from myjive.solver import Constrainer

from fem.jive import CJiveRunner
from probability import Likelihood
from probability.multivariate import Gaussian, SymbolicCovariance
from probability.process import GaussianProcess, ZeroMeanFunction, SquaredExponential
from util.linalg import Matrix

from experiments.inverse.frp_damage.props import get_fem_props
from experiments.inverse.frp_damage import caching, misc, params, sampler

n_burn = 1000
n_sample = 5000
std_pd = 1e-6

h = 0.02

for k in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
    for sigma_e in [1e-4, 1e-5, 1e-6]:
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

        obs_operator = caching.get_or_calc_obs_operator(elems=elems, h=h)
        truth = caching.get_or_calc_true_observations(h=0.002)

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

                self._input_map = (len(domain) - 1) / np.max(domain)
                self._props = get_fem_props()
                self._props["usermodules"]["solver"]["solver"] = {
                    "type": "GMRES",
                    "precision": 1e100,
                }
                self._E_matrix = params.material_params["E_matrix"]

            def calc_logpdf(self, x):
                damage = misc.sigmoid(self.eigenfuncs @ x, 1.0, 0.0)

                for ip, ipoint in enumerate(ipoints):
                    dist = distances[ip]
                    idx_l = int(dist * self._input_map)
                    idx_r = idx_l + 1

                    x_l = domain[idx_l]
                    x_r = domain[idx_r]
                    d_l = damage[idx_l]
                    d_r = damage[idx_r]

                    assert x_l <= dist <= x_r

                    dam = d_l + (dist - x_l) / (x_r - x_l) * (d_r - d_l)
                    backdoor["e"][ip] = self._E_matrix * (1 - dam)

                jive = CJiveRunner(self._props, elems=elems, egroups=egroups)
                globdat = jive(**backdoor)

                K = globdat["matrix0"]
                f = globdat["extForce"]
                c = globdat["constraints"]
                conman = Constrainer(c, K)
                Kc = conman.get_output_matrix()

                Phi = basis[:, :k]
                K_pod = Phi.T @ Kc @ Phi
                f_pod = Phi.T @ f - Phi.T @ K @ lifting

                u_pod = Phi @ np.linalg.solve(K_pod, f_pod) + lifting
                pred = self.operator @ u_pod

                loglikelihood = self.dist.calc_logpdf(pred)
                return loglikelihood

        likelihood = CustomLikelihood()

        def linear_tempering(i):
            if i < n_burn:
                return i / n_burn
            else:
                return 1.0

        ess = sampler.EllipticalSliceSampler(
            prior=kl_prior,
            likelihood=likelihood,
            n_sample=n_sample,
            n_burn=n_burn,
            seed=0,
            tempering=linear_tempering,
            return_info=True,
        )

        samples, info = ess()

        fname = "posterior-samples_pod_h-{:.3f}_noise-{:.0e}_k-{}.npy"
        fname = os.path.join("output", fname.format(h, sigma_e, k))
        np.save(fname, samples)
