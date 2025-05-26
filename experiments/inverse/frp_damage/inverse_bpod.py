import os
import numpy as np
from scipy.sparse import eye_array, diags_array

from myjive.solver import Constrainer

from fem.jive import CJiveRunner
from probability import Likelihood
from probability.multivariate import (
    Gaussian,
    SymbolicCovariance,
    IndependentGaussianSum,
)
from probability.process import GaussianProcess, ZeroMeanFunction, SquaredExponential
from util.linalg import Matrix, MatMulChain

from experiments.inverse.frp_damage.props import get_fem_props
from experiments.inverse.frp_damage import caching, misc, params, sampler

n_burn = 1000
n_sample = 5000
std_pd = 1e-6

h = 0.02
l = 10

for k in [1, 2, 5, 10, 20, 50, 100, 200, 500]:
    for sigma_e in [1e-5]:
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
                self.e = Gaussian(np.zeros(n_obs), self.noise)
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
                Psi = basis[:, : k + l]

                K_phi = Phi.T @ Kc @ Phi
                f_phi = Phi.T @ f - Phi.T @ K @ lifting
                u_phi = np.linalg.solve(K_phi, f_phi)
                u_lifted = Phi @ u_phi + lifting

                K_psi = Psi.T @ Kc @ Psi

                Ix = Matrix(eye_array(Psi.shape[1], Phi.shape[1]), name="Ix")
                Phi = Matrix(Phi, name="Phi")
                Psi = Matrix(Psi, name="Psi")
                K_psi = Matrix(0.5 * (K_psi + K_psi.T), name="K_psi")
                A = Matrix(self.operator, name="A")
                alpha2 = u_phi @ K_phi @ u_phi / len(u_phi)

                cov_prior = SymbolicCovariance(alpha2 * K_psi.inv)
                prior = Gaussian(mean=None, cov=cov_prior)
                operator = MatMulChain(Ix.T, K_psi)
                observations = f_phi
                posterior = prior.condition_on(operator, observations)
                Sigma = posterior.calc_cov()
                Sigma = Matrix(0.5 * (Sigma + Sigma.T), name="Sigma")

                mean = A @ u_lifted
                cov = SymbolicCovariance(A @ Psi @ Sigma @ Psi.T @ A.T)
                post = Gaussian(mean=mean, cov=cov)

                dist = IndependentGaussianSum(post, self.e)
                loglikelihood = dist.calc_logpdf_via_woodbury(self.observations)
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

        fname = "posterior-samples_bpod_h-{:.3f}_noise-{:.0e}_k-{}_l-{}_alpha-{}.npy"
        fname = os.path.join("output", fname.format(h, sigma_e, k, l, "opt"))
        np.save(fname, samples)
