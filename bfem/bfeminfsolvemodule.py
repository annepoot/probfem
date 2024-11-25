import numpy as np
from copy import deepcopy

from myjive.names import GlobNames as gn
from myjive.app import Module
from myjive.solver import Constrainer
from .xconstrainer import xConstrainer
from myjive.util.proputils import split_off_type


class BFEMInfSolveModule(Module):
    @Module.save_config
    def configure(
        self,
        globdat,
        *,
        nsample=0,
        seed=None,
    ):
        self._nsample = nsample
        self._rng = np.random.default_rng(seed)

    def init(self, globdat):
        pass

    def run(self, globdat):
        models = globdat[gn.MODELS]

        globdat["obs"] = {}
        globdat["ref"] = {}

        # Pass the matrices to the prior model
        for model in self.get_relevant_models("RETURNMATRICES", models):
            model.RETURNMATRICES(globdat)

        obsdats = []
        nobs = 0
        for model in self.get_relevant_models("GETOBSDAT", models):
            obsdat = model.GETOBSDAT(globdat)
            obsdats.append(obsdat)
            nobs += obsdat[gn.DOFSPACE].dof_count()

        obsvec = np.zeros(nobs)
        grammat = np.zeros((nobs, nobs))
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

        refdats = []
        for model in self.get_relevant_models("GETREFDAT", models):
            refdat = model.GETREFDAT(globdat)
            refdats.append(refdat)

        for refdat in refdats:
            ref_dof_count = refdat[gn.DOFSPACE].dof_count()
            for obsdat in obsdats:
                obs_dof_count = obsdat[gn.DOFSPACE].dof_count()
                Kx = np.zeros((ref_dof_count, obs_dof_count))
                for model in self.get_relevant_models("GETXMATRIX0", models):
                    model.GETXMATRIX0(Kx, refdat, obsdat)

        for refdat in refdats:
            K = refdat[gn.MATRIX0]
            c = refdat[gn.CONSTRAINTS]

            conman = Constrainer(c, K)
            Kc = conman.get_output_matrix().toarray()

            xconman = xConstrainer(obsdat[gn.CONSTRAINTS], c, Kx)
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

            LS = np.linalg.cholesky(Kc_inv)
            LR = np.linalg.cholesky(G_inv)

            Z = self._rng.standard_normal((len(Kc_inv), self._nsample))
            S_prior = LS @ Z
            S_post = S_prior - Kc_inv @ (Kxc @ (G_inv @ (Kxc.T @ S_prior)))

            # Explicit
            for i in range(5):
                # cov_rev = G_inv - G_inv @ Kxc.T @ Kc_inv @ Kxc @ G_inv
                Y = self._rng.standard_normal((len(G_inv), self._nsample))
                R_post = LR @ Y
                R_post -= G_inv @ (Kxc.T @ (Kc_inv @ (Kxc @ R_post)))
                R_post += G_inv @ Kxc.T @ S_post

                Z = self._rng.standard_normal((len(Kc_inv), self._nsample))
                S_post = LS @ Z
                S_post -= Kc_inv @ (Kxc @ (G_inv @ (Kxc.T @ S_post)))
                S_post += Kc_inv @ (Kxc @ R_post)

            S_post += np.tile(mean, (self._nsample, 1)).T

            refdat["prior"] = {
                "mean": np.zeros_like(mean),
                "cov": Kc_inv,
                "std": np.sqrt(np.diagonal(Kc_inv)),
                "samples": S_prior,
            }
            refdat["posterior"] = {
                "mean": mean,
                "cov": cov,
                "std": np.sqrt(np.diagonal(cov)),
                "samples": S_post,
            }

        return "ok"

    def shutdown(self, globdat):
        pass
