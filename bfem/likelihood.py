import numpy as np

from myjive.names import GlobNames as gn
from myjive.util.proputils import (
    split_key,
    get_recursive,
    set_recursive,
    split_off_type,
)

from probability import Likelihood, FEMObservationOperator
from probability.observation import ObservationOperator
from probability.multivariate import GaussianLike
from probability.process import ProjectedPrior
from bfem.observation import compute_bfem_observations


__all__ = ["BFEMLikelihood"]


class BFEMLikelihood(Likelihood):
    def __init__(self, operator, values, obs_prior, ref_prior, noise):
        assert isinstance(operator, ObservationOperator)
        self.operator = operator
        self.values = values

        self.PhiT = compute_bfem_observations(obs_prior, ref_prior)

        ref_globdat = ref_prior.globdat
        H_obs = self.PhiT @ ref_globdat["matrix0"]
        f_obs = self.PhiT @ ref_globdat["extForce"]

        posterior = ref_prior.condition_on(H_obs, f_obs)
        self.posterior = posterior.to_gaussian(allow_singular=True)

        assert isinstance(noise, GaussianLike)
        self.noise = noise

    def calc_pdf(self, x):
        prediction = self.operator.calc_prediction(x)
        de = (prediction + self.noise).to_gaussian()
        return de.calc_pdf(self.values)

    def calc_logpdf(self, x):
        prediction = self.operator.calc_prediction(x)
        de = (prediction + self.noise).to_gaussian()
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
        run_modules,
    ):
        assert isinstance(obs_prior, ProjectedPrior)
        assert isinstance(ref_prior, ProjectedPrior)
        self.obs_prior = obs_prior
        self.ref_prior = ref_prior

        self.input_variables = input_variables
        self.output_locations = output_locations
        self.output_dofs = output_dofs
        self.run_modules = run_modules

    def calc_prediction(self, x):
        if len(x) != len(self.input_variables):
            raise ValueError

        modelprops = {}
        for var in self.input_variables:
            name = split_key(var)[0]
            model = self.obs_prior.globdat[gn.MODELS][name]
            modelprops[name] = split_off_type(model.get_config())[1]

        for xi, var in zip(x, self.input_variables):
            keys = split_key(var)
            assert get_recursive(modelprops, keys) is not None
            set_recursive(modelprops, keys, xi)

        obsdat = self.obs_prior.globdat
        refdat = self.ref_prior.globdat

        for name, config in modelprops.items():
            obsdat[gn.MODELS][name].configure(obsdat, **config)
            refdat[gn.MODELS][name].configure(refdat, **config)

        for name in self.run_modules:
            obsmodule = obsdat[gn.MODULES][name]
            refmodule = refdat[gn.MODULES][name]

            obsmodule.run(obsdat)
            refmodule.run(refdat)

        PhiT = compute_bfem_observations(self.obs_prior, self.ref_prior)
        H_obs = PhiT @ refdat["matrix0"]
        f_obs = PhiT @ refdat["extForce"]
        posterior = self.ref_prior.condition_on(H_obs, f_obs)
        self.posterior = posterior.to_gaussian(allow_singular=True)

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

        return self.posterior @ mapper.T
