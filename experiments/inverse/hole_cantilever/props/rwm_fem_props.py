import numpy as np
import pandas as pd

from probability.univariate import Uniform
from probability.multivariate import IsotropicGaussian
from probability import (
    ProportionalPosterior,
    Likelihood,
    RemeshFEMObservationOperator,
    IndependentJoint,
)
from fem.jive import CJiveRunner

from experiments.inverse.hole_cantilever.meshing import create_mesh

__all__ = ["get_rwm_fem_target"]


def get_rwm_fem_target(*, h, h_meas, std_corruption, sigma_e):
    df = pd.read_csv("ground-truth.csv", skiprows=10)
    ground_locs = df[["loc_x", "loc_y"]].to_numpy()
    ground_dofs = df["dof_type"].to_numpy()
    ground_truth = df["value"].to_numpy()

    # generate repeated noisy observations
    n_obs = len(ground_truth)
    rng = np.random.default_rng(0)
    corruption = std_corruption * rng.standard_normal(n_obs)
    obs_vals = ground_truth + corruption
    obs_locs = ground_locs
    obs_dofs = ground_dofs

    node_count_estimate = int(1000 / h)

    jive_runner = CJiveRunner(
        props="props/fem.pro",
        node_count=node_count_estimate,
        elem_count=2 * node_count_estimate,
        rank=2,
        max_elem_node_count=3,
    )

    mesh_props = {
        "h": h,
        "L": 4,
        "H": 1,
        "x": 2,
        "y": 0.5,
        "a": 0.4,
        "theta": 0.0,
        "r_rel": 0.0,
        "h_meas": h_meas,
        "fname": "cantilever.msh",
    }

    obs_op = RemeshFEMObservationOperator(
        jive_runner=jive_runner,
        mesher=create_mesh,
        mesh_props=mesh_props,
        input_variables=["x", "y", "a", "theta", "r_rel"],
        output_locations=obs_locs,
        output_dofs=obs_dofs,
    )

    target = ProportionalPosterior(
        prior=IndependentJoint(
            Uniform(0.5, 4.0),
            Uniform(0.0, 1.0),
            Uniform(0.0, 0.5),
            Uniform(0.0, 2 * np.pi),
            Uniform(0.0, 0.5),
        ),
        likelihood=Likelihood(
            operator=obs_op,
            values=obs_vals,
            noise=IsotropicGaussian(mean=None, std=sigma_e, size=n_obs),
        ),
    )

    return target
