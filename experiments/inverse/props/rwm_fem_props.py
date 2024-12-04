import numpy as np

from .fem_props import get_fem_props
from probability.multivariate import IsotropicGaussian
from probability import (
    ProportionalPosterior,
    Likelihood,
    FEMObservationOperator,
)


def get_rwm_fem_props(*, std_corruption, sigma_e, n_rep_obs):
    fem_props = get_fem_props()

    # ground truth (9 locations, 9 true values)
    ground_locations = np.linspace(0, 1, 11)[1:-1].reshape((-1, 1))
    ground_truth = np.array(
        [
            0.01154101,
            0.01667733,
            0.01592942,
            0.00980423,
            -0.00043005,
            -0.01177105,
            -0.02001336,
            -0.0211289,
            -0.01350695,
        ]
    )

    # generate repeated noisy observations
    n_obs = len(ground_truth) * n_rep_obs
    rng = np.random.default_rng(0)
    corruption = std_corruption * rng.standard_normal(n_obs)
    obs_values = np.tile(ground_truth, n_rep_obs) + corruption
    obs_locations = np.tile(ground_locations, (n_rep_obs, 1))

    rwm_fem_props = {
        "target": {
            "type": ProportionalPosterior,
            "prior": {"type": IsotropicGaussian, "mean": np.zeros(4), "std": 1},
            "likelihood": {
                "type": Likelihood,
                "operator": {
                    "type": FEMObservationOperator,
                    "input_variables": [
                        "solid.material.params.xi_1",
                        "solid.material.params.xi_2",
                        "solid.material.params.xi_3",
                        "solid.material.params.xi_4",
                    ],
                    "output_type": "local",
                    "output_variables": ["state0"] * n_obs,
                    "output_locations": obs_locations,
                    "output_dofs": ["dx"] * n_obs,
                    "forward_props": fem_props,
                    "run_modules": ["solve"],
                },
                "values": obs_values,  # tbd
                "noise": {
                    "type": IsotropicGaussian,
                    "mean": np.zeros(n_obs),
                    "std": sigma_e,
                },
            },
        },
        "proposal": {"type": IsotropicGaussian, "mean": np.zeros(4), "std": 1},
        "nsample": 10000,
        "startValue": np.zeros(4),
        "seed": 0,
    }

    return rwm_fem_props
