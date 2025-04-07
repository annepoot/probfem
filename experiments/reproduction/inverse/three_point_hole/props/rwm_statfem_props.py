import numpy as np

from probability import ParametrizedLikelihood, IndependentJoint, RejectConditional
from probability.univariate import Gaussian
from probability.process import GaussianProcess, SquaredExponential
from statfem.likelihood import StatFEMLikelihood

from experiments.reproduction.inverse.three_point_hole.props import get_rwm_fem_target

__all__ = ["get_rwm_statfem_target"]


def get_rwm_statfem_target(*, h, h_meas, std_corruption, sigma_e, folder=""):
    target = get_rwm_fem_target(
        h=h,
        h_meas=h_meas,
        std_corruption=std_corruption,
        sigma_e=sigma_e,
        folder=folder,
    )

    old_prior = target.prior
    old_joint = old_prior.latent.distributions
    log_rho_prior = Gaussian(np.log(1), 0.1)
    log_l_d_prior = Gaussian(np.log(1), np.log(1e1))
    log_sigma_d_prior = Gaussian(np.log(1e-4), np.log(1e1))
    joint_prior = IndependentJoint(
        *old_joint, log_rho_prior, log_l_d_prior, log_sigma_d_prior
    )
    target.prior = RejectConditional(latent=joint_prior, reject_if=old_prior.reject_if)

    old_likelihood = target.likelihood
    new_likelihood = ParametrizedLikelihood(
        likelihood=StatFEMLikelihood(
            operator=old_likelihood.operator,
            values=old_likelihood.values,
            rho=1.0,  # temp
            d=GaussianProcess(
                mean=None, cov=SquaredExponential(l=1.0, sigma=1.0)
            ),  # temp
            e=old_likelihood.noise,
        ),
        hyperparameters=["rho", "d.cov.l", "d.cov.sigma"],
        transforms=[np.exp, np.exp, np.exp],
    )
    target.likelihood = new_likelihood

    return target
