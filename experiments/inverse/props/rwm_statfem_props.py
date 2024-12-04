import numpy as np

from .rwm_fem_props import get_rwm_fem_props
from probability.multivariate import IsotropicGaussian
from probability.process import GaussianProcess, SquaredExponential
from statfem.likelihood import StatFEMLikelihood


def get_rwm_statfem_props(*, std_corruption, rho, l_d, sigma_d, sigma_e, n_rep_obs):
    rwm_statfem_props = get_rwm_fem_props(
        std_corruption=std_corruption, sigma_e=sigma_e, n_rep_obs=n_rep_obs
    )

    likelihood_props = rwm_statfem_props["target"]["likelihood"]
    likelihood_props["type"] = StatFEMLikelihood
    likelihood_props["rho"] = rho
    likelihood_props["d"] = {
        "type": GaussianProcess,
        "mean": None,
        "cov": {"type": SquaredExponential, "l": l_d, "sigma": sigma_d},
    }
    likelihood_props["e"] = likelihood_props.pop("noise")
    likelihood_props["e"]["std"] = sigma_e
    obs_locations = likelihood_props["operator"]["output_locations"].flatten()
    likelihood_props["locations"] = obs_locations

    return rwm_statfem_props
