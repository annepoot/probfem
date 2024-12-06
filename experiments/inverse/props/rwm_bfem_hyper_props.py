import numpy as np

from probability.univariate import LogGaussian
from probability import ParametrizedLikelihood, IndependentJoint
from .rwm_bfem_props import get_rwm_bfem_target


def get_rwm_bfem_hyper_target(*, n_elem, std_corruption, sigma_e, n_rep_obs):
    target = get_rwm_bfem_target(
        n_elem=n_elem,
        std_corruption=std_corruption,
        scale=1.0,
        sigma_e=sigma_e,
        n_rep_obs=n_rep_obs,
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
