import numpy as np

from myjive.names import GlobNames as gn

from probability.process import (
    ProjectedPrior,
    InverseCovarianceOperator,
    NaturalCovarianceOperator,
    ZeroMeanFunction,
)
from meshing import create_phi_from_globdat

__all__ = [
    "compute_bfem_observations",
    "compute_cg_observations",
    "compute_random_observations",
]


def compute_bfem_observations(coarse_prior, fine_prior, *, fspace=False):
    assert isinstance(coarse_prior, ProjectedPrior)
    inf_prior = coarse_prior.prior
    assert isinstance(inf_prior.mean, ZeroMeanFunction)
    assert isinstance(
        inf_prior.cov, (InverseCovarianceOperator, NaturalCovarianceOperator)
    )
    coarse_globdat = coarse_prior.globdat

    assert isinstance(fine_prior, ProjectedPrior)
    inf_prior = fine_prior.prior
    assert isinstance(inf_prior.mean, ZeroMeanFunction)
    assert isinstance(
        inf_prior.cov, (InverseCovarianceOperator, NaturalCovarianceOperator)
    )
    fine_globdat = fine_prior.globdat

    Phi = create_phi_from_globdat(coarse_globdat, fine_globdat)
    c = coarse_globdat[gn.CONSTRAINTS]

    cdofs = c.get_constraints()[0]
    Phic = np.delete(Phi, cdofs, axis=1)

    return Phic.T


from myjive.solver.cgsolver import CGSolver


def compute_cg_observations(
    matrix, rhs, constraints, *, renormalize, n_obs=None, solver_props={}
):
    solver = CGSolver("CG")
    solver.configure({}, **solver_props)

    solver.update(matrix, constraints)

    lhs = np.zeros_like(rhs)
    if n_obs is None:
        n_obs = len(rhs)

    PhiT = np.zeros((n_obs, len(rhs)))
    PhiT[0] = solver.get_residual(lhs, rhs)

    for i in range(1, n_obs):
        res = solver.get_residual(lhs, rhs)
        dx = solver.iterate(res)

        lhs += dx

        if renormalize:
            res = -solver.get_residual(lhs, rhs)
            mat = solver._matrix

            p = res
            for phi in PhiT[:i]:
                p -= ((phi @ mat @ res) / (phi @ mat @ phi)) * phi

            assert np.allclose(solver._p, p)
            solver._p = p
        else:
            p = solver._p

        PhiT[i] = p

    return PhiT


def compute_random_observations(n_dof, *, renormalize=True, n_obs=None, seed=None):
    if n_obs is None:
        n_obs = n_dof

    PhiT = np.zeros((n_obs, n_dof))
    rng = np.random.default_rng(seed)

    for i in range(n_obs):
        p = rng.standard_normal(n_dof)

        if renormalize:
            newp = p
            for phi in PhiT[:i]:
                newp -= ((phi @ p) / (phi @ phi)) * phi
            p = newp

        PhiT[i] = p

    return PhiT
