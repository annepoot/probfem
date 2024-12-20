from rmfem import PseudoMarginalLikelihood, RMFEMObservationOperator

from experiments.inverse.kl_bar.props.rwm_fem_props import get_rwm_fem_target

__all__ = ["get_rwm_rmfem_target"]


def get_rwm_rmfem_target(
    *, n_elem, std_corruption, sigma_e, n_rep_obs, n_pseudomarginal
):
    target = get_rwm_fem_target(
        n_elem=n_elem,
        std_corruption=std_corruption,
        sigma_e=sigma_e,
        n_rep_obs=n_rep_obs,
    )

    old_likelihood = target.likelihood

    old_operator = old_likelihood.operator
    new_operator = RMFEMObservationOperator(
        p=1,
        seed=0,
        forward_props=old_operator.forward_props,
        input_variables=old_operator.input_variables,
        output_type=old_operator.output_type,
        output_variables=old_operator.output_variables,
        output_locations=old_operator.output_locations,
        output_dofs=old_operator.output_dofs,
        run_modules=old_operator.run_modules,
    )
    old_likelihood.operator = new_operator

    new_likelihood = PseudoMarginalLikelihood(
        likelihood=old_likelihood, n_sample=n_pseudomarginal
    )
    target.likelihood = new_likelihood

    return target
