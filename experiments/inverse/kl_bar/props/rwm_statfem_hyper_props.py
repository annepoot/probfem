import numpy as np

from probability.univariate import LogGaussian
from probability import ParametrizedLikelihood, IndependentJoint

from experiments.inverse.kl_bar.props.rwm_statfem_props import get_rwm_statfem_target

__all__ = ["get_rwm_statfem_hyper_target"]


def get_rwm_statfem_hyper_target(*, n_elem, std_corruption, n_rep_obs, sigma_e):
    target = get_rwm_statfem_target(
        n_elem=n_elem,
        std_corruption=std_corruption,
        rho=1.0,
        l_d=1.0,
        sigma_d=1.0,
        sigma_e=sigma_e,
        n_rep_obs=n_rep_obs,
    )
    param_prior = target.prior
    rho_prior = LogGaussian(np.log(1), np.log(1e1), allow_logscale_access=True)
    l_d_prior = LogGaussian(np.log(1), np.log(1e1), allow_logscale_access=True)
    sigma_d_prior = LogGaussian(np.log(1e-4), np.log(1e1), allow_logscale_access=True)
    joint_prior = IndependentJoint(param_prior, rho_prior, l_d_prior, sigma_d_prior)
    target.prior = joint_prior

    old_likelihood = target.likelihood
    new_likelihood = ParametrizedLikelihood(
        likelihood=old_likelihood,
        hyperparameters=["rho", "d.cov.l", "d.cov.sigma"],
    )
    target.likelihood = new_likelihood

    return target
