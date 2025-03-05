import numpy as np

from probability.univariate import Uniform
from probability.multivariate import IsotropicGaussian
from probability import (
    TemperedPosterior,
    Likelihood,
    RemeshFEMObservationOperator,
    IndependentJoint,
)
from fem.jive import CJiveRunner
from util.io import read_csv_from

from experiments.inverse.three_point_hole.meshing import create_mesh
from experiments.inverse.three_point_hole.props import get_fem_props

__all__ = ["get_rwm_fem_target"]


def get_rwm_fem_target(*, h, h_meas, std_corruption, sigma_e):
    fem_props = get_fem_props()

    # ground truth
    df = read_csv_from("ground-truth.csv", "loc_x,loc_y,inode,dof_idx,dof_type")
    ground_locs = df[["loc_x", "loc_y"]].to_numpy()
    ground_dofs = df["dof_type"].to_numpy()
    ground_truth = df["value"].to_numpy()

    # generate repeated noisy observations
    n_obs = len(ground_truth)
    rng = np.random.default_rng(0)
    corruption = std_corruption * rng.standard_normal(n_obs)
    obs_values = ground_truth + corruption
    obs_locs = ground_locs
    obs_dofs = ground_dofs

    mesh_props = {
        "h": h,
        "L": 5.0,
        "H": 1.0,
        "U": 0.5,
        "x": 2.5,
        "y": 0.5,
        "a": 0.5,
        "theta": 0.0,
        "r_rel": 0.25,
        "h_meas": h_meas,
    }

    nodes, elems = create_mesh(**mesh_props)

    jive = CJiveRunner(props=fem_props, elems=elems)

    target = TemperedPosterior(
        prior=IndependentJoint(
            Uniform(0.0, 5.0),
            Uniform(0.0, 1.0),
            Uniform(0.0, 0.5),
            Uniform(0.0, 2 * np.pi),
            Uniform(0.0, 0.5),
        ),
        likelihood=Likelihood(
            operator=RemeshFEMObservationOperator(
                jive_runner=jive,
                mesher=create_mesh,
                mesh_props=mesh_props,
                input_variables=["x", "y", "a", "theta", "r_rel"],
                output_locations=obs_locs,
                output_dofs=obs_dofs,
                mandatory_coords=np.array(
                    [
                        [0.4, 0.0],
                        [0.5, -0.1],
                        [0.6, 0.0],
                        [2.4, 1.0],
                        [2.5, 1.1],
                        [2.6, 1.0],
                        [4.4, 0.0],
                        [4.5, -0.1],
                        [4.6, 0.0],
                    ]
                ),
            ),
            values=obs_values,
            noise=IsotropicGaussian(mean=None, std=sigma_e, size=n_obs),
        ),
    )

    return target
