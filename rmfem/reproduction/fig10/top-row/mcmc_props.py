import numpy as np

from fem_props import fem_props
from probability import (
    Gaussian,
    ProportionalPosterior,
    Likelihood,
    FEMObservationOperator,
)

prop_cov = 1e-4 * np.identity(4)

mcmc_props = {
    "target": {
        "type": ProportionalPosterior,
        "prior": {
            "type": Gaussian,
            "mean": np.zeros(4),
            "cov": np.identity(4),
        },
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
                "output_variables": ["state0"] * 9,
                "output_locations": np.linspace(0, 1, 11)[1:-1].reshape((-1, 1)),
                "output_dofs": ["dx"] * 9,
                "forward_props": fem_props,
                "run_modules": ["solve"],
            },
            "values": None,  # tbd
            "noise": {
                "type": Gaussian,
                "mean": None,
                "cov": None,  # tbd
            },
        },
    },
    "proposal": {"type": Gaussian, "mean": None, "cov": 1e-4 * np.identity(4)},
    "nsample": 10000,
    "startValue": np.array([1.0, 1.0, 0.25, 0.25]),
    "seed": 0,
}
