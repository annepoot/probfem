import numpy as np

from probability import ParametrizedLikelihood, IndependentJoint
from probability.univariate import LogGaussian, Gaussian
from probability.process import GaussianProcess, SquaredExponential
from statfem.likelihood import StatFEMLikelihood

from experiments.reproduction.inverse.pullout_bar.props import get_rwm_fem_target

__all__ = ["get_rwm_statfem_target"]


def get_rwm_statfem_target(*, elems, std_corruption, sigma_e):
    target = get_rwm_fem_target(
        elems=elems,
        std_corruption=std_corruption,
        sigma_e=sigma_e,
    )

    old_prior = target.prior.distributions
    log_rho_prior = Gaussian(np.log(1), 0.1)
    log_l_d_prior = Gaussian(np.log(1), np.log(1e1))
    log_sigma_d_prior = Gaussian(np.log(1e-4), np.log(1e1))
    joint_prior = IndependentJoint(
        *old_prior, log_rho_prior, log_l_d_prior, log_sigma_d_prior
    )
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
        transforms=[np.exp, np.exp, np.exp],
    )
    target.likelihood = new_likelihood

    return target
