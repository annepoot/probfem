from copy import deepcopy

from probability.process import (
    GaussianProcess,
    InverseCovarianceOperator,
    ProjectedPrior,
)
from bfem.likelihood import BFEMLikelihood, RemeshBFEMObservationOperator
from fem.jive import CJiveRunner

from experiments.inverse.hole_cantilever.props.rwm_fem_props import get_rwm_fem_target

__all__ = ["get_rwm_bfem_target"]


def get_rwm_bfem_target(*, h, h_meas, std_corruption, scale, rescale, sigma_e):
    target = get_rwm_fem_target(
        h=h, h_meas=h_meas, std_corruption=std_corruption, sigma_e=sigma_e
    )

    # not verified yet
    assert rescale == False

    if rescale:
        assert abs(1.0 - scale) < 1e-8

    obs_module_props = target.likelihood.operator.jive_runner.props
    ref_module_props = deepcopy(obs_module_props)
    ref_module_props["userinput"]["gmsh"]["file"] = "cantilever-r1.msh"
    obs_model_props = obs_module_props.pop("model")
    ref_model_props = ref_module_props.pop("model")

    assert obs_model_props == ref_model_props

    inf_cov = InverseCovarianceOperator(model_props=ref_model_props, scale=scale)
    inf_prior = GaussianProcess(None, inf_cov)

    node_buffer_size = int(1000 / h)
    elem_buffer_size = 2 * node_buffer_size

    obs_jive_kws = {
        "node_count": node_buffer_size,
        "elem_count": elem_buffer_size,
        "rank": 2,
        "max_elem_node_count": 3,
    }
    ref_jive_kws = {
        "node_count": 4 * node_buffer_size,
        "elem_count": 4 * elem_buffer_size,
        "rank": 2,
        "max_elem_node_count": 3,
    }

    obs_prior = ProjectedPrior(
        prior=inf_prior,
        module_props=obs_module_props,
        jive_runner=CJiveRunner,
        jive_runner_kws=obs_jive_kws,
    )
    ref_prior = ProjectedPrior(
        prior=inf_prior,
        module_props=ref_module_props,
        jive_runner=CJiveRunner,
        jive_runner_kws=ref_jive_kws,
    )

    old_likelihood = target.likelihood
    new_likelihood = BFEMLikelihood(
        operator=old_likelihood.operator,
        values=old_likelihood.values,
        noise=old_likelihood.noise,
    )
    target.likelihood = new_likelihood

    old_operator = target.likelihood.operator
    mesh_props = old_operator.mesh_props
    assert "n_refine" not in mesh_props
    mesh_props["n_refine"] = 1

    new_operator = RemeshBFEMObservationOperator(
        mesher=old_operator.mesher,
        mesh_props=mesh_props,
        obs_prior=obs_prior,
        ref_prior=ref_prior,
        input_variables=old_operator.input_variables,
        output_locations=old_operator.output_locations,
        output_dofs=old_operator.output_dofs,
        rescale=rescale,
    )
    target.likelihood.operator = new_operator

    return target
