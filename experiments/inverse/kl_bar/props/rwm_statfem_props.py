from probability.process import GaussianProcess, SquaredExponential
from statfem.likelihood import StatFEMLikelihood

from experiments.inverse.kl_bar.props.rwm_fem_props import get_rwm_fem_target

__all__ = ["get_rwm_statfem_target"]


def get_rwm_statfem_target(*, elems, std_corruption, rho, l_d, sigma_d, sigma_e):
    target = get_rwm_fem_target(
        elems=elems,
        std_corruption=std_corruption,
        sigma_e=sigma_e,
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
