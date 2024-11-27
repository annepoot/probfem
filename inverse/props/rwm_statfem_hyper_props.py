import numpy as np

from .rwm_statfem_props import get_rwm_statfem_props
from probability.univariate import Gaussian
from probability import IndependentJoint
from statfem.likelihood import ParametrizedStatFEMLikelihood


def get_rwm_statfem_hyper_props(*, std_corruption, n_rep_obs):
    rwm_statfem_hyper_props = get_rwm_statfem_props(
        std_corruption=std_corruption,
        rho=1.0,
        l_d=1.0,
        sigma_d=1.0,
        sigma_e=1.0,
        n_rep_obs=n_rep_obs,
    )

    rwm_statfem_hyper_props["target"]["prior"] = {
        "type": IndependentJoint,
        "distributions": [
            rwm_statfem_hyper_props["target"].pop("prior"),
            {"type": Gaussian, "mean": np.log(1), "std": np.log(1e1)},
            {"type": Gaussian, "mean": np.log(1), "std": np.log(1e1)},
            {"type": Gaussian, "mean": np.log(1e-4), "std": np.log(1e1)},
            {"type": Gaussian, "mean": np.log(1e-4), "std": np.log(1e1)},
        ],
    }
    rwm_statfem_hyper_props["target"]["likelihood"][
        "type"
    ] = ParametrizedStatFEMLikelihood
    rwm_statfem_hyper_props["target"]["likelihood"]["hyperparameters"] = [
        "rho",
        "d.cov.l",
        "d.cov.sigma",
        "e.std",
    ]
    rwm_statfem_hyper_props["proposal"] = {
        "type": IndependentJoint,
        "distributions": [
            rwm_statfem_hyper_props.pop("proposal"),
            {"type": Gaussian, "mean": np.log(1), "std": np.log(1e1)},
            {"type": Gaussian, "mean": np.log(1), "std": np.log(1e1)},
            {"type": Gaussian, "mean": np.log(1e-4), "std": np.log(1e1)},
            {"type": Gaussian, "mean": np.log(1e-4), "std": np.log(1e1)},
        ],
    }
    rwm_statfem_hyper_props["startValue"] = np.zeros(8)

    return rwm_statfem_hyper_props
