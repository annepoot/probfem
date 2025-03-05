import numpy as np

from fem.jive import CJiveRunner
from probability.multivariate import IsotropicGaussian
from probability.joint import IndependentJoint
from probability.univariate import LogGaussian
from probability import TemperedPosterior, Likelihood, FEMObservationOperator

from experiments.inverse.pullout_bar.props.fem_props import get_fem_props

__all__ = ["get_rwm_fem_target"]


def get_rwm_fem_target(*, elems, std_corruption, sigma_e):
    fem_props = get_fem_props()

    # ground truth (9 locations, 9 true values)
    k = fem_props["model"]["model"]["spring"]["k"]
    E = fem_props["model"]["model"]["elastic"]["material"]["E"]
    f = fem_props["model"]["model"]["neum"]["initLoad"]
    nu = np.sqrt(k / E)
    eps = f / E

    u_exact = eps * (np.exp(nu) + np.exp(-nu)) / (nu * (np.exp(nu) - np.exp(-nu)))
    ground_truth = u_exact

    # generate repeated noisy observations
    rng = np.random.default_rng(0)
    corruption = std_corruption * rng.standard_normal()
    obs_value = ground_truth + corruption
    obs_inode = len(elems)

    jive = CJiveRunner(props=fem_props, elems=elems)

    target = TemperedPosterior(
        prior=IndependentJoint(
            LogGaussian(np.log(1.0), 0.1, allow_logscale_access=True),
            LogGaussian(np.log(100.0), 0.1, allow_logscale_access=True),
        ),
        likelihood=Likelihood(
            operator=FEMObservationOperator(
                jive_runner=jive,
                input_variables=[
                    "model.model.elastic.material.E",
                    "model.model.spring.k",
                ],
                output_type="nodal",
                output_variables=["state0"],
                output_locations=np.array([obs_inode]),
                output_dofs=["dx"],
            ),
            values=obs_value,
            noise=IsotropicGaussian(mean=None, std=sigma_e, size=1),
        ),
    )

    return target
