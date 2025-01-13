from rmfem import PseudoMarginalLikelihood, RemeshRMFEMObservationOperator
from experiments.inverse.hole_cantilever.props.rwm_fem_props import get_rwm_fem_target

__all__ = ["get_rwm_rmfem_target"]


def get_rwm_rmfem_target(*, h, h_meas, std_corruption, sigma_e, n_pseudomarginal):
    target = get_rwm_fem_target(
        h=h, h_meas=h_meas, std_corruption=std_corruption, sigma_e=sigma_e
    )

    n_meas = int(9 / h_meas) + 1

    old_likelihood = target.likelihood

    old_operator = old_likelihood.operator
    new_operator = RemeshRMFEMObservationOperator(
        p=1,
        seed=0,
        omit_nodes=range(0, n_meas),
        jive_runner=old_operator.jive_runner,
        mesher=old_operator.mesher,
        mesh_props=old_operator.mesh_props,
        input_variables=old_operator.input_variables,
        output_locations=old_operator.output_locations,
        output_dofs=old_operator.output_dofs,
    )
    old_likelihood.operator = new_operator

    new_likelihood = PseudoMarginalLikelihood(
        likelihood=old_likelihood, n_sample=n_pseudomarginal
    )
    target.likelihood = new_likelihood

    return target
