from probability.process import (
    GaussianProcess,
    InverseCovarianceOperator,
    ProjectedPrior,
)
from bfem.likelihood import BFEMLikelihood, BFEMObservationOperator
from .rwm_fem_props import get_rwm_fem_target
from .fem_props import get_fem_props


def get_rwm_bfem_target(*, n_elem, std_corruption, scale, rescale, sigma_e, n_rep_obs):
    target = get_rwm_fem_target(
        n_elem=n_elem,
        std_corruption=std_corruption,
        sigma_e=sigma_e,
        n_rep_obs=n_rep_obs,
    )

    if rescale:
        assert abs(1.0 - scale) < 1e-8

    obs_props = get_fem_props(n_elem=n_elem)
    ref_props = get_fem_props(n_elem=80)

    inf_cov = InverseCovarianceOperator(model_props=ref_props["model"], scale=scale)
    inf_prior = GaussianProcess(None, inf_cov)
    obs_prior = ProjectedPrior(
        prior=inf_prior, init_props=obs_props["init"], solve_props=obs_props["solve"]
    )
    ref_prior = ProjectedPrior(
        prior=inf_prior, init_props=ref_props["init"], solve_props=ref_props["solve"]
    )

    old_likelihood = target.likelihood
    new_likelihood = BFEMLikelihood(
        operator=old_likelihood.operator,
        values=old_likelihood.values,
        noise=old_likelihood.noise,
    )
    target.likelihood = new_likelihood

    old_operator = target.likelihood.operator
    new_operator = BFEMObservationOperator(
        obs_prior=obs_prior,
        ref_prior=ref_prior,
        input_variables=old_operator.input_variables,
        output_locations=old_operator.output_locations,
        output_dofs=old_operator.output_dofs,
        run_modules=old_operator.run_modules,
        rescale=rescale,
    )
    target.likelihood.operator = new_operator

    return target
