import numpy as np
from scipy.sparse import csr_array
from warnings import warn

from myjive.names import GlobNames as gn
from myjive.util.proputils import split_key, get_recursive, set_recursive

from probability import Likelihood, FEMObservationOperator, RemeshFEMObservationOperator
from probability.observation import ObservationOperator
from probability.multivariate import GaussianLike, IndependentGaussianSum, Gaussian
from probability.process import ProjectedPrior
from bfem.observation import compute_bfem_observations


__all__ = ["BFEMLikelihood"]


class BFEMLikelihood(Likelihood):
    def __init__(self, operator, values, noise):
        assert isinstance(operator, ObservationOperator)
        self.operator = operator
        self.values = values

        assert isinstance(noise, GaussianLike)
        self.noise = noise

    def calc_pdf(self, x):
        prediction = self.operator.calc_prediction(x)
        if np.isnan(np.sum(prediction.calc_mean())):
            return -np.inf
        else:
            de = (prediction + self.noise).to_gaussian()
            return de.calc_pdf(self.values)

    def calc_logpdf(self, x):
        prediction = self.operator.calc_prediction(x)
        if np.isnan(np.sum(prediction.calc_mean())):
            return -np.inf
        else:
            de = IndependentGaussianSum(prediction, self.noise)
            return de.calc_logpdf(self.values)


class BFEMObservationOperator(FEMObservationOperator):
    def __init__(
        self,
        *,
        obs_prior,
        ref_prior,
        input_variables,
        output_locations,
        output_dofs,
        rescale,
    ):
        assert isinstance(obs_prior, ProjectedPrior)
        assert isinstance(ref_prior, ProjectedPrior)
        self.obs_prior = obs_prior
        self.ref_prior = ref_prior

        self.input_variables = input_variables
        self.output_locations = output_locations
        self.output_dofs = output_dofs
        self.rescale = rescale

    def calc_prediction(self, x):
        if len(x) != len(self.input_variables):
            raise ValueError

        for x_i, var in zip(x, self.input_variables):
            keys = split_key(var)
            assert get_recursive(self.obs_prior.jive_runner.props, keys) is not None
            set_recursive(self.obs_prior.jive_runner.props, keys, x_i)
            assert get_recursive(self.ref_prior.jive_runner.props, keys) is not None
            set_recursive(self.ref_prior.jive_runner.props, keys, x_i)

        self.ref_prior.recompute_moments()
        self.obs_prior.recompute_moments()

        refdat = self.ref_prior.globdat

        H_obs, f_obs = compute_bfem_observations(self.obs_prior, self.ref_prior)
        posterior = self.ref_prior.condition_on(H_obs, f_obs)

        if self.rescale:
            oldmean = posterior.calc_mean()
            oldcov = posterior.calc_cov()
            l, Q = np.linalg.eigh(oldcov)
            newl = l * abs(Q.T @ refdat["extForce"])
            newcov = Q @ np.diag(newl) @ Q.T
            self.posterior = Gaussian(oldmean, newcov, allow_singular=True)
        else:
            self.posterior = posterior

        n_out = len(self.output_locations)
        assert len(self.output_dofs) == n_out

        idx = np.where(
            np.sum(
                np.subtract.outer(self.output_locations, refdat[gn.NSET].get_coords())
                ** 2,
                axis=(1, 3),
            )
            < 1e-8
        )
        assert np.all(idx[0] == np.arange(n_out))

        mapper = np.zeros((n_out, refdat[gn.DOFSPACE].dof_count()))
        for i, (inode, dof) in enumerate(zip(idx[1], self.output_dofs)):
            idof = refdat[gn.DOFSPACE].get_dof(inode, dof)
            mapper[i, idof] = 1.0

        mapper = csr_array(mapper)

        return self.posterior @ mapper.T


class RemeshBFEMObservationOperator(RemeshFEMObservationOperator):
    def __init__(
        self,
        *,
        mesher,
        mesh_props,
        obs_prior,
        ref_prior,
        input_variables,
        output_locations,
        output_dofs,
        rescale,
    ):
        self.mesher = mesher
        self.mesh_props = mesh_props

        assert isinstance(obs_prior, ProjectedPrior)
        assert isinstance(ref_prior, ProjectedPrior)
        self.obs_prior = obs_prior
        self.ref_prior = ref_prior

        self.input_variables = input_variables
        self.output_locations = output_locations
        self.output_dofs = output_dofs
        self.rescale = rescale

    def calc_prediction(self, x):
        if len(x) != len(self.input_variables):
            raise ValueError

        for x_i, var in zip(x, self.input_variables):
            assert var in self.mesh_props
            self.mesh_props[var] = x_i

        meshes = self.mesher(**self.mesh_props)
        obs_nodes, obs_elems = meshes[0]
        ref_nodes, ref_elems = meshes[1]

        self.obs_prior.jive_runner.update_elems(obs_elems)
        self.ref_prior.jive_runner.update_elems(ref_elems)

        self.obs_prior.recompute_moments()
        self.ref_prior.recompute_moments()

        refdat = self.ref_prior.globdat

        H_obs, f_obs = compute_bfem_observations(self.obs_prior, self.ref_prior)
        posterior = self.ref_prior.condition_on(H_obs, f_obs)

        if self.rescale:
            oldmean = posterior.calc_mean()
            oldcov = posterior.calc_cov()
            l, Q = np.linalg.eigh(oldcov)
            newl = l * abs(Q.T @ refdat["extForce"])
            newcov = Q @ np.diag(newl) @ Q.T
            self.posterior = Gaussian(oldmean, newcov, allow_singular=True)
        else:
            self.posterior = posterior

        n_out = len(self.output_locations)
        assert len(self.output_dofs) == n_out
        ref_coords = refdat[gn.NSET].get_coords()
        ref_dofs = refdat[gn.DOFSPACE]

        tol = 1e-8

        mapper = np.zeros((n_out, ref_dofs.dof_count()))

        for i, (loc, dof) in enumerate(zip(self.output_locations, self.output_dofs)):
            inodes = np.where(np.all(abs(ref_coords - loc) < tol, axis=1))[0]
            if len(inodes) == 0:
                warn("Observation location not found. Getting closest node instead")
                inode = np.argmin(np.sum((ref_coords - loc) ** 2, axis=1))
                idof = ref_dofs.get_dof(inode, dof)
                mapper[i, idof] = np.nan
            elif len(inodes) == 1:
                inode = inodes[0]
                idof = ref_dofs.get_dof(inode, dof)
                mapper[i, idof] = 1.0
            else:
                assert False

        mapper = csr_array(mapper)

        return self.posterior @ mapper.T
