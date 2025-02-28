from probability.process import (
    GaussianProcess,
    InverseCovarianceOperator,
    ProjectedPrior,
)
from bfem.likelihood import BFEMLikelihood, RemeshBFEMObservationOperator
from fem.jive import CJiveRunner

from experiments.inverse.cantilever_hole.props import get_fem_props, get_rwm_fem_target

__all__ = ["get_rwm_bfem_target"]


def get_rwm_bfem_target(*, h, h_meas, std_corruption, scale, sigma_e):
    target = get_rwm_fem_target(
        h=h,
        h_meas=h_meas,
        std_corruption=std_corruption,
        sigma_e=sigma_e,
    )

    if scale in ["mle", "eig"]:
        rescale = scale
        scale = 1.0
    else:
        rescale = None

    obs_module_props = get_fem_props()
    ref_module_props = get_fem_props()

    obs_model_props = obs_module_props.pop("model")
    ref_model_props = ref_module_props.pop("model")

    assert obs_model_props == ref_model_props

    mesher = target.likelihood.operator.mesher
    mesh_props = target.likelihood.operator.mesh_props
    assert "n_refine" not in mesh_props
    mesh_props["n_refine"] = 1

    meshes = mesher(**mesh_props)
    obs_nodes, obs_elems = meshes[0]
    ref_nodes, ref_elems = meshes[1]

    obs_jive_runner = CJiveRunner(obs_module_props, elems=obs_elems)
    ref_jive_runner = CJiveRunner(ref_module_props, elems=ref_elems)

    inf_cov = InverseCovarianceOperator(model_props=ref_model_props, scale=scale)
    inf_prior = GaussianProcess(None, inf_cov)

    obs_prior = ProjectedPrior(prior=inf_prior, jive_runner=obs_jive_runner)
    ref_prior = ProjectedPrior(prior=inf_prior, jive_runner=ref_jive_runner)

    old_likelihood = target.likelihood
    new_likelihood = BFEMLikelihood(
        operator=old_likelihood.operator,
        values=old_likelihood.values,
        noise=old_likelihood.noise,
    )
    target.likelihood = new_likelihood

    old_operator = target.likelihood.operator
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
