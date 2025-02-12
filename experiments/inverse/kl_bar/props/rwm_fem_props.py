import numpy as np

from fem.jive import CJiveRunner
from probability.multivariate import IsotropicGaussian
from probability import TemperedPosterior, Likelihood, FEMObservationOperator

from experiments.inverse.kl_bar.props.fem_props import get_fem_props

__all__ = ["get_rwm_fem_target"]


def get_rwm_fem_target(*, elems, std_corruption, sigma_e):
    fem_props = get_fem_props()

    # ground truth (9 locations, 9 true values)
    n_loc = 9
    ground_locations = np.linspace(0, 1, n_loc + 2)[1:-1].reshape((-1, 1))

    if n_loc == 4:
        ground_truth = np.array(
            [
                0.01667733,
                0.00980423,
                -0.01177105,
                -0.0211289,
            ]
        )
    elif n_loc == 9:
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
    else:
        raise ValueError

    # generate repeated noisy observations
    n_obs = len(ground_truth)
    rng = np.random.default_rng(0)
    corruption = std_corruption * rng.standard_normal(n_obs)
    obs_values = ground_truth + corruption
    obs_locations = ground_locations

    jive = CJiveRunner(props=fem_props, elems=elems)

    target = TemperedPosterior(
        prior=IsotropicGaussian(mean=None, std=1, size=4),
        likelihood=Likelihood(
            operator=FEMObservationOperator(
                jive_runner=jive,
                input_variables=[
                    "model.model.elastic.material.params.values.0",
                    "model.model.elastic.material.params.values.1",
                    "model.model.elastic.material.params.values.2",
                    "model.model.elastic.material.params.values.3",
                ],
                output_type="local",
                output_variables=["state0"] * n_obs,
                output_locations=obs_locations,
                output_dofs=["dx"] * n_obs,
            ),
            values=obs_values,
            noise=IsotropicGaussian(mean=None, std=sigma_e, size=n_obs),
        ),
    )

    return target
