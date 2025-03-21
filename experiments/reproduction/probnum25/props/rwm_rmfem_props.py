import numpy as np

from rmfem import PseudoMarginalLikelihood, RMFEMObservationOperator

from experiments.reproduction.probnum25.props.rwm_fem_props import get_rwm_fem_target

__all__ = ["get_rwm_rmfem_target"]


def get_rwm_rmfem_target(
    *, elems, std_corruption, sigma_e, n_pseudomarginal, omit_nodes
):
    target = get_rwm_fem_target(
        elems=elems,
        std_corruption=std_corruption,
        sigma_e=sigma_e,
    )

    old_likelihood = target.likelihood
    old_operator = old_likelihood.operator

    assert isinstance(omit_nodes, bool)
    if omit_nodes:
        omit_coords = old_operator.output_locations
    else:
        omit_coords = np.zeros((0, 1))

    new_operator = RMFEMObservationOperator(
        p=1,
        seed=0,
        jive_runner=old_operator.jive_runner,
        input_variables=old_operator.input_variables,
        input_transforms=old_operator.input_transforms,
        output_type=old_operator.output_type,
        output_variables=old_operator.output_variables,
        output_locations=old_operator.output_locations,
        output_dofs=old_operator.output_dofs,
        omit_coords=omit_coords,
    )
    old_likelihood.operator = new_operator

    new_likelihood = PseudoMarginalLikelihood(
        likelihood=old_likelihood, n_sample=n_pseudomarginal
    )
    target.likelihood = new_likelihood

    return target
