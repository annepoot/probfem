import numpy as np
from scipy.sparse import eye_array

from myjive.solver import Constrainer

from bfem.observation import compute_bfem_observations
from fem.jive import CJiveRunner
from fem.meshing import create_phi_from_globdat
from probability import Likelihood
from probability.multivariate import (
    Gaussian,
    SymbolicCovariance,
    IndependentGaussianSum,
)
from probability.process import (
    GaussianProcess,
    InverseCovarianceOperator,
    ProjectedPrior,
)
from util.linalg import Matrix, MatMulChain

from experiments.inverse.frp_damage.props import get_fem_props
from experiments.inverse.frp_damage import misc, params


class FEMLikelihood(Likelihood):

    def __init__(
        self,
        *,
        operator,
        observations,
        sigma_e,
        ipoints,
        distances,
        eigenfuncs,
        domain,
        egroups,
        backdoor,
    ):
        self.ipoints = ipoints
        self.distances = distances
        self.operator = operator
        self.observations = observations
        n_obs = len(self.observations)
        self.noise = SymbolicCovariance(Matrix(sigma_e**2 * eye_array(n_obs)))
        self.dist = Gaussian(self.observations, self.noise)
        self.eigenfuncs = eigenfuncs
        self.egroups = egroups
        self.elems = next(iter(egroups.values())).get_elements()
        self.backdoor = backdoor

        self._props = get_fem_props()
        self._E_matrix = params.material_params["E_matrix"]
        self._damage_map = misc.calc_damage_map(ipoints, distances, domain)

    def calc_logpdf(self, x):
        damage = misc.sigmoid(self.eigenfuncs @ x, 1.0, 0.0)
        self.backdoor["e"] = self._E_matrix * (1 - self._damage_map @ damage)

        jive = CJiveRunner(self._props, elems=self.elems, egroups=self.egroups)
        globdat = jive(**self.backdoor)

        state0 = globdat["state0"]
        pred = self.operator @ state0

        loglikelihood = self.dist.calc_logpdf(pred)
        return loglikelihood


class PODLikelihood(Likelihood):

    def __init__(
        self,
        *,
        operator,
        observations,
        sigma_e,
        basis,
        k,
        lifting,
        ipoints,
        distances,
        eigenfuncs,
        domain,
        egroups,
        backdoor,
    ):
        self.ipoints = ipoints
        self.distances = distances
        self.operator = operator
        self.observations = observations
        n_obs = len(self.observations)
        self.noise = SymbolicCovariance(Matrix(sigma_e**2 * eye_array(n_obs)))
        self.basis = basis
        self.k = k
        self.lifting = lifting
        self.dist = Gaussian(self.observations, self.noise)
        self.eigenfuncs = eigenfuncs
        self.egroups = egroups
        self.elems = next(iter(egroups.values())).get_elements()
        self.backdoor = backdoor

        self._props = get_fem_props()
        self._props["usermodules"]["solver"]["solver"] = {
            "type": "GMRES",
            "precision": 1e100,
        }
        self._E_matrix = params.material_params["E_matrix"]
        self._damage_map = misc.calc_damage_map(ipoints, distances, domain)
        self._Phi = self.basis[:, : self.k]

    def calc_logpdf(self, x):
        damage = misc.sigmoid(self.eigenfuncs @ x, 1.0, 0.0)
        self.backdoor["e"] = self._E_matrix * (1 - self._damage_map @ damage)

        jive = CJiveRunner(self._props, elems=self.elems, egroups=self.egroups)
        globdat = jive(**self.backdoor)

        K = globdat["matrix0"]
        f = globdat["extForce"]
        c = globdat["constraints"]
        conman = Constrainer(c, K)
        Kc = conman.get_output_matrix()

        K_pod = self._Phi.T @ Kc @ self._Phi
        f_pod = self._Phi.T @ f - self._Phi.T @ K @ self.lifting

        u_pod = self._Phi @ np.linalg.solve(K_pod, f_pod) + self.lifting
        pred = self.operator @ u_pod

        loglikelihood = self.dist.calc_logpdf(pred)
        return loglikelihood


class BFEMLikelihoodHierarchical(Likelihood):

    def __init__(
        self,
        *,
        obs_operator,
        ref_operator,
        observations,
        sigma_e,
        obs_ipoints,
        ref_ipoints,
        obs_distances,
        ref_distances,
        eigenfuncs,
        domain,
        obs_egroups,
        ref_egroups,
        obs_backdoor,
        ref_backdoor,
    ):
        self.obs_ipoints = obs_ipoints
        self.obs_distances = obs_distances
        self.obs_operator = obs_operator

        self.ref_ipoints = ref_ipoints
        self.ref_distances = ref_distances
        self.ref_operator = ref_operator

        self.observations = observations
        n_obs = len(self.observations)
        self.noise = SymbolicCovariance(Matrix(sigma_e**2 * eye_array(n_obs)))
        self.e = Gaussian(np.zeros(n_obs), self.noise)
        self.eigenfuncs = eigenfuncs
        self.obs_egroups = obs_egroups
        self.ref_egroups = ref_egroups
        self.obs_elems = next(iter(obs_egroups.values())).get_elements()
        self.ref_elems = next(iter(ref_egroups.values())).get_elements()
        self.obs_backdoor = obs_backdoor
        self.ref_backdoor = ref_backdoor

        self._input_map = (len(domain) - 1) / np.max(domain)
        self._module_props = get_fem_props()
        self._module_props["usermodules"]["solver"]["solver"] = {
            "type": "GMRES",
            "precision": 1e100,
        }
        self._model_props = self._module_props.pop("model")
        self._E_matrix = params.material_params["E_matrix"]
        self._damage_map_obs = misc.calc_damage_map(
            self.obs_ipoints, self.obs_distances, domain
        )
        self._damage_map_ref = misc.calc_damage_map(
            self.ref_ipoints, self.ref_distances, domain
        )

        self._Phi = None

    def calc_logpdf(self, x):
        damage = misc.sigmoid(self.eigenfuncs @ x, 1.0, 0.0)
        damage_obs = self._damage_map_obs @ damage
        damage_ref = self._damage_map_ref @ damage
        self.obs_backdoor["e"] = self._E_matrix * (1 - damage_obs)
        self.ref_backdoor["e"] = self._E_matrix * (1 - damage_ref)

        obs_jive_runner = CJiveRunner(
            self._module_props, elems=self.obs_elems, egroups=self.obs_egroups
        )
        ref_jive_runner = CJiveRunner(
            self._module_props, elems=self.ref_elems, egroups=self.ref_egroups
        )

        inf_cov = InverseCovarianceOperator(model_props=self._model_props, scale=1.0)
        inf_prior = GaussianProcess(None, inf_cov)

        obs_prior = ProjectedPrior(
            prior=inf_prior, jive_runner=obs_jive_runner, **self.obs_backdoor
        )
        ref_prior = ProjectedPrior(
            prior=inf_prior, jive_runner=ref_jive_runner, **self.ref_backdoor
        )

        obsdat = obs_prior.globdat
        refdat = ref_prior.globdat

        u_obs = obsdat["state0"]
        K_obs = obsdat["matrix0"]
        n_obs = len(u_obs)
        alpha2_mle = u_obs @ K_obs @ u_obs / n_obs

        assert ref_prior.prior.cov.scale == 1.0
        ref_prior.prior.cov.scale = alpha2_mle
        assert obs_prior.prior.cov.scale == alpha2_mle

        obs_prior.recompute_moments(**self.obs_backdoor)
        ref_prior.recompute_moments(**self.ref_backdoor)

        if self._Phi is None:
            self._Phi = create_phi_from_globdat(obsdat, refdat)

        H_obs, f_obs = compute_bfem_observations(obs_prior, ref_prior, Phi=self._Phi)
        posterior = ref_prior.condition_on(H_obs, f_obs)
        prediction = posterior @ self.ref_operator.T

        dist = IndependentGaussianSum(prediction, self.e)
        loglikelihood = dist.calc_logpdf_via_woodbury(self.observations)
        return loglikelihood


class BPODLikelihoodHierarchical(Likelihood):

    def __init__(
        self,
        *,
        operator,
        observations,
        sigma_e,
        basis,
        k,
        l,
        lifting,
        ipoints,
        distances,
        eigenfuncs,
        domain,
        egroups,
        backdoor,
    ):
        self.ipoints = ipoints
        self.distances = distances
        self.operator = operator
        self.observations = observations
        n_obs = len(self.observations)
        self.noise = SymbolicCovariance(Matrix(sigma_e**2 * eye_array(n_obs)))
        self.e = Gaussian(np.zeros(n_obs), self.noise)
        self.basis = basis
        self.k = k
        self.l = l
        self.lifting = lifting
        self.eigenfuncs = eigenfuncs
        self.egroups = egroups
        self.elems = next(iter(egroups.values())).get_elements()
        self.backdoor = backdoor

        self._props = get_fem_props()
        self._props["usermodules"]["solver"]["solver"] = {
            "type": "GMRES",
            "precision": 1e100,
        }
        self._E_matrix = params.material_params["E_matrix"]
        self._damage_map = misc.calc_damage_map(ipoints, distances, domain)
        self._Phi = self.basis[:, : self.k]
        self._Psi = self.basis[:, : self.k + self.l]

    def calc_logpdf(self, x):
        damage = misc.sigmoid(self.eigenfuncs @ x, 1.0, 0.0)
        self.backdoor["e"] = self._E_matrix * (1 - self._damage_map @ damage)

        jive = CJiveRunner(self._props, elems=self.elems, egroups=self.egroups)
        globdat = jive(**self.backdoor)

        K = globdat["matrix0"]
        f = globdat["extForce"]
        c = globdat["constraints"]
        conman = Constrainer(c, K)
        Kc = conman.get_output_matrix()

        K_phi = self._Phi.T @ Kc @ self._Phi
        f_phi = self._Phi.T @ f - self._Phi.T @ K @ self.lifting
        u_phi = np.linalg.solve(K_phi, f_phi)
        u_lifted = self._Phi @ u_phi + self.lifting

        K_psi = self._Psi.T @ Kc @ self._Psi

        Ix = Matrix(eye_array(self._Psi.shape[1], self._Phi.shape[1]), name="Ix")
        Phi = Matrix(self._Phi, name="Phi")
        Psi = Matrix(self._Psi, name="Psi")
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


class BPODLikelihoodHeterarchical(Likelihood):

    def __init__(
        self,
        *,
        operator,
        observations,
        sigma_e,
        basis,
        k,
        l,
        lifting,
        ipoints,
        distances,
        eigenfuncs,
        domain,
        egroups,
        backdoor,
    ):
        self.ipoints = ipoints
        self.distances = distances
        self.operator = operator
        self.observations = observations
        n_obs = len(self.observations)
        self.noise = SymbolicCovariance(Matrix(sigma_e**2 * eye_array(n_obs)))
        self.e = Gaussian(np.zeros(n_obs), self.noise)
        self.basis = basis
        self.k = k
        self.l = l
        self.lifting = lifting
        self.eigenfuncs = eigenfuncs
        self.egroups = egroups
        self.elems = next(iter(egroups.values())).get_elements()
        self.backdoor = backdoor

        self._props = get_fem_props()
        self._props["usermodules"]["solver"]["solver"] = {
            "type": "GMRES",
            "precision": 1e100,
        }
        self._E_matrix = params.material_params["E_matrix"]
        self._damage_map = misc.calc_damage_map(ipoints, distances, domain)
        self._Phi = self.basis[:, : self.k]
        self._Psi = self.basis[:, self.k : self.l]

    def calc_logpdf(self, x):
        damage = misc.sigmoid(self.eigenfuncs @ x, 1.0, 0.0)
        self.backdoor["e"] = self._E_matrix * (1 - self._damage_map @ damage)

        jive = CJiveRunner(self._props, elems=self.elems, egroups=self.egroups)
        globdat = jive(**self.backdoor)

        K = globdat["matrix0"]
        f = globdat["extForce"]
        c = globdat["constraints"]
        conman = Constrainer(c, K)
        Kc = conman.get_output_matrix()
        Kc = Matrix(0.5 * (Kc + Kc.T), name="K")

        Phi = Matrix(self._Phi, name="Phi")
        Psi = Matrix(self._Psi, name="Psi")
        K_phi = Matrix((Phi.T @ Kc @ Phi).evaluate(), name="K_phi")
        K_psi = Matrix((Psi.T @ Kc @ Psi).evaluate(), name="K_psi")
        K_x = Matrix((Psi.T @ Kc @ Phi).evaluate(), name="K_x")

        f_phi = Phi.T @ f - Phi.T @ K @ self.lifting
        u_phi = K_phi.inv @ f_phi
        u_lifted = Phi @ u_phi + self.lifting

        A = Matrix(self.operator, name="A")
        alpha2 = u_phi @ (K_phi @ u_phi) / len(u_phi)

        Sigma = K_psi.inv.evaluate()
        downdate = K_psi.inv @ K_x @ K_phi.inv @ K_x.T @ K_psi.inv
        Sigma -= downdate.evaluate()
        Sigma = Matrix(0.5 * alpha2 * (Sigma + Sigma.T), name="Sigma")

        mean = A @ u_lifted
        cov = SymbolicCovariance(A @ Psi @ Sigma @ Psi.T @ A.T)
        post = Gaussian(mean=mean, cov=cov)

        dist = IndependentGaussianSum(post, self.e)
        loglikelihood = dist.calc_logpdf_via_woodbury(self.observations)
        return loglikelihood
