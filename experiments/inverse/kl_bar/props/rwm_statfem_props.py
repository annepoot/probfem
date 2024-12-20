from probability.process import GaussianProcess, SquaredExponential
from statfem.likelihood import StatFEMLikelihood

from experiments.inverse.kl_bar.props.rwm_fem_props import get_rwm_fem_target

__all__ = ["get_rwm_statfem_target"]


def get_rwm_statfem_target(
    *, n_elem, std_corruption, rho, l_d, sigma_d, sigma_e, n_rep_obs
):
    target = get_rwm_fem_target(
        n_elem=n_elem,
        std_corruption=std_corruption,
        sigma_e=sigma_e,
        n_rep_obs=n_rep_obs,
    )

    old_likelihood = target.likelihood
    new_likelihood = StatFEMLikelihood(
        operator=old_likelihood.operator,
        values=old_likelihood.values,
        rho=rho,
        d=GaussianProcess(mean=None, cov=SquaredExponential(l=l_d, sigma=sigma_d)),
        e=old_likelihood.noise,
        locations=old_likelihood.operator.output_locations.flatten(),
    )
    target.likelihood = new_likelihood

    return target
