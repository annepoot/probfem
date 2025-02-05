import numpy as np

from probability.univariate import LogGaussian
from probability import ParametrizedLikelihood, IndependentJoint

from experiments.inverse.kl_bar.props.rwm_bfem_props import get_rwm_bfem_target

__all__ = ["get_rwm_bfem_hyper_target"]


def get_rwm_bfem_hyper_target(
    *, obs_elems, ref_elems, std_corruption, rescale, sigma_e
):
    target = get_rwm_bfem_target(
        obs_elems=obs_elems,
        ref_elems=ref_elems,
        std_corruption=std_corruption,
        scale=1.0,
        rescale=rescale,
        sigma_e=sigma_e,
    )

    param_prior = target.prior
    hyper_prior = LogGaussian(np.log(1e0), np.log(1e1), allow_logscale_access=True)
    joint_prior = IndependentJoint(param_prior, hyper_prior)
    target.prior = joint_prior

    old_likelihood = target.likelihood
    new_likelihood = ParametrizedLikelihood(
        likelihood=old_likelihood,
        hyperparameters=["operator.ref_prior.prior.cov.scale"],
    )
    target.likelihood = new_likelihood

    return target
