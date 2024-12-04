import numpy as np

from myjive.names import GlobNames as gn
from myjive.solver import Constrainer
from myjive.util.proputils import split_off_type

from bfem.xsolidmodel import XSolidModel
from bfem.xconstrainer import XConstrainer
from probability.multivariate import Gaussian
from probability.process import (
    ZeroMeanFunction,
    InverseCovarianceOperator,
    ProjectedPrior,
)


__all__ = ["compute_nonhierarchical_posterior"]


def compute_nonhierarchical_posterior(obs_priors, ref_prior):
    obsdats = []
    if isinstance(obs_priors, ProjectedPrior):
        obs_priors = [obs_priors]
    assert isinstance(obs_priors, list)
    for obs_prior in obs_priors:
        assert isinstance(obs_prior, ProjectedPrior)
        inf_prior = obs_prior.prior
        assert isinstance(inf_prior.mean, ZeroMeanFunction)
        assert isinstance(inf_prior.cov, InverseCovarianceOperator)
        obsdats.append(obs_prior.globdat)

    assert isinstance(ref_prior, ProjectedPrior)
    inf_prior = ref_prior.prior
    assert isinstance(inf_prior.mean, ZeroMeanFunction)
    assert isinstance(inf_prior.cov, InverseCovarianceOperator)
    refdat = ref_prior.globdat

    n_obs = 0
    for obs_prior in obs_priors:
        n_obs += obs_prior.globdat[gn.DOFSPACE].dof_count()

    obsvec = np.zeros(n_obs)
    grammat = np.zeros((n_obs, n_obs))
    offset = 0
    for obsdat in obsdats:
        nobs = obsdat[gn.DOFSPACE].dof_count()
        K = obsdat[gn.MATRIX0]
        f = obsdat[gn.EXTFORCE]
        c = obsdat[gn.CONSTRAINTS]

        conman = Constrainer(c, K)
        Kc = conman.get_output_matrix()
        fc = conman.get_rhs(f)

        grammat[offset : offset + nobs, offset : offset + nobs] += Kc
        obsvec[offset : offset + nobs] += fc

    config = split_off_type(refdat[gn.MODELS]["solid"].get_config())[1]
    xsolid = XSolidModel("xsolid")
    xsolid.configure(refdat, **config)

    ref_dof_count = refdat[gn.DOFSPACE].dof_count()
    assert len(obsdats) == 1
    for obsdat in obsdats:
        obs_dof_count = obsdat[gn.DOFSPACE].dof_count()
        Kx = np.zeros((ref_dof_count, obs_dof_count))
        xsolid.GETXMATRIX0(Kx, refdat, obsdat)

        K = refdat[gn.MATRIX0]
        c = refdat[gn.CONSTRAINTS]

        conman = Constrainer(c, K)
        Kc = conman.get_output_matrix().toarray()

        xconman = XConstrainer(obsdat[gn.CONSTRAINTS], c, Kx)
        Kxc = xconman.get_output_matrix().toarray()

        Kc_inv = np.linalg.inv(Kc)
        G_inv = np.linalg.inv(grammat)

        cobs = obsdat[gn.CONSTRAINTS].get_constraints()[0]
        cref = refdat[gn.CONSTRAINTS].get_constraints()[0]

        G_inv[np.ix_(cobs, cobs)] *= 0.0
        G_inv[cobs, cobs] += 1e-8
        Kxc[np.ix_(cref, cobs)] *= 0.0
        Kc_inv[np.ix_(cref, cref)] *= 0.0
        Kc_inv[cref, cref] += 1e-8

        mean = Kc_inv @ (Kxc @ (G_inv @ obsvec))
        cov = Kc_inv - Kc_inv @ Kxc @ G_inv @ Kxc.T @ Kc_inv.T

        posterior = Gaussian(mean, cov, allow_singular=True)

        # LS = np.linalg.cholesky(Kc_inv)
        # LR = np.linalg.cholesky(G_inv)

        # rng = np.random.default_rng(seed)

        # Z = rng.standard_normal((len(Kc_inv), n_sample))
        # S_prior = LS @ Z
        # S_post = S_prior - Kc_inv @ (Kxc @ (G_inv @ (Kxc.T @ S_prior)))

        # # Explicit
        # for i in range(5):
        #     # cov_rev = G_inv - G_inv @ Kxc.T @ Kc_inv @ Kxc @ G_inv
        #     Y = rng.standard_normal((len(G_inv), n_sample))
        #     R_post = LR @ Y
        #     R_post -= G_inv @ (Kxc.T @ (Kc_inv @ (Kxc @ R_post)))
        #     R_post += G_inv @ Kxc.T @ S_post

        #     Z = rng.standard_normal((len(Kc_inv), n_sample))
        #     S_post = LS @ Z
        #     S_post -= Kc_inv @ (Kxc @ (G_inv @ (Kxc.T @ S_post)))
        #     S_post += Kc_inv @ (Kxc @ R_post)

        # S_post += np.tile(mean, (n_sample, 1)).T

        # refdat["prior"] = {
        #     "mean": np.zeros_like(mean),
        #     "cov": Kc_inv,
        #     "std": np.sqrt(np.diagonal(Kc_inv)),
        #     "samples": S_prior,
        # }
        # refdat["posterior"] = {
        #     "mean": mean,
        #     "cov": cov,
        #     "std": np.sqrt(np.diagonal(cov)),
        #     "samples": S_post,
        # }

    return posterior
