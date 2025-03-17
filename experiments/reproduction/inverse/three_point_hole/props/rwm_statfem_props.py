import numpy as np

from probability import ParametrizedLikelihood, IndependentJoint
from probability.univariate import LogGaussian
from probability.process import GaussianProcess, SquaredExponential
from statfem.likelihood import StatFEMLikelihood

from experiments.reproduction.inverse.three_point_hole.props import get_rwm_fem_target

__all__ = ["get_rwm_statfem_target"]


def get_rwm_statfem_target(*, h, h_meas, std_corruption, sigma_e):
    target = get_rwm_fem_target(
        h=h,
        h_meas=h_meas,
        std_corruption=std_corruption,
        sigma_e=sigma_e,
    )

    old_prior = target.prior.distributions
    rho_prior = LogGaussian(np.log(1), 0.1, allow_logscale_access=True)
    l_d_prior = LogGaussian(np.log(1), np.log(1e1), allow_logscale_access=True)
    sigma_d_prior = LogGaussian(np.log(1e-4), np.log(1e1), allow_logscale_access=True)
    joint_prior = IndependentJoint(*old_prior, rho_prior, l_d_prior, sigma_d_prior)
    target.prior = joint_prior

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
    )
    target.likelihood = new_likelihood

    return target
