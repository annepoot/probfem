from rmfem import PseudoMarginalLikelihood, RMFEMObservationOperator

from experiments.reproduction.probnum25.props.rwm_fem_2d_props import get_rwm_fem_2d_target

__all__ = ["get_rwm_rmfem_2d_target"]


def get_rwm_rmfem_2d_target(
    *, elems, std_corruption, sigma_e, n_pseudomarginal, omit_nodes
):
    target = get_rwm_fem_2d_target(
        elems=elems,
        std_corruption=std_corruption,
        sigma_e=sigma_e,
    )

    old_likelihood = target.likelihood

    assert isinstance(omit_nodes, bool)
    if omit_nodes:
        n_obs = len(old_likelihood.values)
        n_elem = len(elems)
        assert n_elem % (n_obs + 1) == 0
        omit_nodes_list = [i * n_elem // (n_obs + 1) for i in range(1, n_obs + 1)]
    else:
        omit_nodes_list = []

    old_operator = old_likelihood.operator
    new_operator = RMFEMObservationOperator(
        p=1,
        seed=0,
        jive_runner=old_operator.jive_runner,
        input_variables=old_operator.input_variables,
        output_type=old_operator.output_type,
        output_variables=old_operator.output_variables,
        output_locations=old_operator.output_locations,
        output_dofs=old_operator.output_dofs,
        omit_nodes=omit_nodes_list,
    )
    old_likelihood.operator = new_operator

    new_likelihood = PseudoMarginalLikelihood(
        likelihood=old_likelihood, n_sample=n_pseudomarginal
    )
    target.likelihood = new_likelihood

    return target
