import numpy as np

from .rwm_fem_props import get_rwm_fem_props
from probability.univariate import Gaussian
from probability import ParametrizedLikelihood, IndependentJoint


def get_rwm_fem_hyper_props(*, n_elem, std_corruption, n_rep_obs):
    rwm_fem_hyper_props = get_rwm_fem_props(
        n_elem=n_elem, std_corruption=std_corruption, sigma_e=1.0, n_rep_obs=n_rep_obs
    )

    rwm_fem_hyper_props["target"]["prior"] = {
        "type": IndependentJoint,
        "distributions": [
            rwm_fem_hyper_props["target"].pop("prior"),
            {"type": Gaussian, "mean": np.log(1e-4), "std": np.log(1e1)},
        ],
    }
    rwm_fem_hyper_props["target"]["likelihood"]["type"] = ParametrizedLikelihood
    rwm_fem_hyper_props["target"]["likelihood"]["hyperparameters"] = ["noise.std"]
    rwm_fem_hyper_props["proposal"] = {
        "type": IndependentJoint,
        "distributions": [
            rwm_fem_hyper_props.pop("proposal"),
            {"type": Gaussian, "mean": np.log(1e-4), "std": np.log(1e1)},
        ],
    }
    rwm_fem_hyper_props["startValue"] = np.zeros(5)

    return rwm_fem_hyper_props
