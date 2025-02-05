import numpy as np

from probability.univariate import LogGaussian
from probability import ParametrizedLikelihood, IndependentJoint

from experiments.inverse.kl_bar.props.rwm_fem_props import get_rwm_fem_target

__all__ = ["get_rwm_fem_hyper_target"]


def get_rwm_fem_hyper_target(*, elems, std_corruption):
    target = get_rwm_fem_target(elems=elems, std_corruption=std_corruption, sigma_e=1.0)

    param_prior = target.prior
    noise_prior = LogGaussian(np.log(1e-4), np.log(1e1), allow_logscale_access=True)
    joint_prior = IndependentJoint(param_prior, noise_prior)
    target.prior = joint_prior

    old_likelihood = target.likelihood
    new_likelihood = ParametrizedLikelihood(
        likelihood=old_likelihood,
        hyperparameters=["noise.update_std"],
    )
    target.likelihood = new_likelihood

    return target
