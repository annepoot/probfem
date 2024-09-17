import numpy as np
from copy import deepcopy

from myjive.names import GlobNames as gn
from myjive.app import Module
from myjive.solver import Constraints
from myjive.util.proputils import split_off_type


class BFEMSolveModule(Module):
    @Module.save_config
    def configure(
        self,
        globdat,
        *,
        fineSolve={"type": "Linsolve"},
        sequential,
        nsample=0,
    ):

        solvetype, solveprops = split_off_type(fineSolve)

        self._fine_solve = globdat[gn.MODULEFACTORY].get_module(solvetype, "fineSolve")
        self._fine_solve.configure(globdat, **solveprops)

        self._sequential = sequential
        self._nsample = nsample

    def init(self, globdat):
        pass

    def run(self, globdat):
        self._fine_solve.solve(globdat)

        models = globdat[gn.MODELS]
        fglobdat = globdat
        fmodels = fglobdat[gn.MODELS]

        # Figure out which matrices to get from the fine-scale models
        needed = []
        for model in self.get_relevant_models("ASKMATRICES", models):
            needed.extend(model.ASKMATRICES())

        for matrix in np.unique(needed):
            if matrix == "K":
                K = self._fine_solve._get_empty_matrix(fglobdat)
                for model in self.get_relevant_models("GETMATRIX0", fmodels):
                    K = model.GETMATRIX0(K, fglobdat)
                globdat[gn.MATRIX0] = K
            elif matrix == "M":
                M = self._fine_solve._get_empty_matrix(fglobdat)
                for model in self.get_relevant_models("GETMATRIX2", fmodels):
                    M = model.GETMATRIX2(M, fglobdat, unit_matrix=True)
                globdat[gn.MATRIX2] = M
            else:
                raise NotImplementedError(
                    "not sure what to make of this type of matrix"
                )

        c = Constraints()
        for model in self.get_relevant_models("GETCONSTRAINTS", fmodels):
            c = model.GETCONSTRAINTS(c, fglobdat)
        globdat[gn.CONSTRAINTS] = c

        globdat["obs"] = {}

        # Pass the matrices to the prior model
        for model in self.get_relevant_models("RETURNMATRICES", models):
            model.RETURNMATRICES(globdat)

        model = self.get_unique_relevant_model("GETPRIOR", models)
        prior = model.GETPRIOR(globdat)

        prior = prior.to_direct_gaussian()

        sequence = []

        for model in self.get_relevant_models("APPLYPOSTTRANS", models):
            sequence.append(model.APPLYPOSTTRANS(prior, globdat))

        posterior = deepcopy(prior)

        for model in self.get_relevant_models("GETOBSERVATIONS", models):

            Phi, measurements, noise = model.GETOBSERVATIONS(globdat)

            name = model.get_name()
            if name not in globdat["obs"]:
                globdat["obs"][name] = {}

            globdat["obs"][name]["Phi"] = Phi
            globdat["obs"][name]["measurements"] = measurements

            if isinstance(self._sequential, bool):
                if self._sequential:
                    for phi, measurement in zip(Phi.T, measurements):
                        posterior = posterior.condition_on(phi, measurement, noise)
                        for model in self.get_relevant_models("APPLYPOSTTRANS", models):
                            sequence.append(model.APPLYPOSTTRANS(posterior, globdat))
                else:
                    posterior = posterior.condition_on(Phi.T, measurements, noise)
                    for model in self.get_relevant_models("APPLYPOSTTRANS", models):
                        sequence.append(model.APPLYPOSTTRANS(posterior, globdat))
            elif isinstance(self._sequential, int):
                block = self._sequential
                for i in range(Phi.shape[1] // block + 1):
                    phiT = Phi.T[i * block : (i + 1) * block]
                    meas = measurements[i * block : (i + 1) * block]
                    posterior = posterior.condition_on(phiT, meas, noise)
                    for model in self.get_relevant_models("APPLYPOSTTRANS", models):
                        sequence.append(model.APPLYPOSTTRANS(posterior, globdat))

        # Create a dictionary for the gp output
        prior = sequence[0]
        posterior = sequence[-1]

        globdat["gp"] = {}
        globdat["gp"]["sequence"] = sequence
        globdat["gp"]["mean"] = {}
        globdat["gp"]["mean"]["prior"] = {}
        globdat["gp"]["mean"]["prior"]["state0"] = prior.calc_mean()
        globdat["gp"]["mean"]["prior"]["extForce"] = prior._latent.calc_mean()
        globdat["gp"]["mean"]["posterior"] = {}
        globdat["gp"]["mean"]["posterior"]["state0"] = posterior.calc_mean()
        globdat["gp"]["mean"]["posterior"]["extForce"] = posterior._latent.calc_mean()
        globdat["gp"]["cov"] = {}
        globdat["gp"]["cov"]["prior"] = {}
        globdat["gp"]["cov"]["prior"]["state0"] = prior.calc_cov()
        globdat["gp"]["cov"]["prior"]["extForce"] = prior._latent.calc_cov()
        globdat["gp"]["cov"]["posterior"] = {}
        globdat["gp"]["cov"]["posterior"]["state0"] = posterior.calc_cov()
        globdat["gp"]["cov"]["posterior"]["extForce"] = posterior._latent.calc_cov()
        globdat["gp"]["std"] = {}
        globdat["gp"]["std"]["prior"] = {}
        globdat["gp"]["std"]["prior"]["state0"] = prior.calc_std()
        globdat["gp"]["std"]["prior"]["extForce"] = prior._latent.calc_std()
        globdat["gp"]["std"]["posterior"] = {}
        globdat["gp"]["std"]["posterior"]["state0"] = posterior.calc_std()
        globdat["gp"]["std"]["posterior"]["extForce"] = posterior._latent.calc_std()

        if self._nsample > 0:
            globdat["gp"]["samples"] = {}
            globdat["gp"]["samples"]["prior"] = {}
            globdat["gp"]["samples"]["prior"]["state0"] = prior.calc_samples(
                self._nsample, 0
            )
            globdat["gp"]["samples"]["prior"]["extForce"] = prior._latent.calc_samples(
                self._nsample, 0
            )
            globdat["gp"]["samples"]["posterior"] = {}
            globdat["gp"]["samples"]["posterior"]["state0"] = posterior.calc_samples(
                self._nsample, 0
            )
            globdat["gp"]["samples"]["posterior"]["extForce"] = (
                posterior._latent.calc_samples(self._nsample, 0)
            )

        return "ok"

    def shutdown(self, globdat):
        pass
