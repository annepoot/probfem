import numpy as np
from copy import deepcopy

from myjive.names import GlobNames as gn
from myjive.app import Module
from myjive.implicit import LinsolveModule
from myjive.solver import Constraints
from myjive.util.proputils import split_off_type

from bfem import BFEMObservationModel, BoundaryObservationModel


class BFEMSolveModule(Module):
    @LinsolveModule.save_config
    def configure(
        self,
        globdat,
        *,
        coarseSolve={"type": "Linsolve"},
        fineSolve={"type": "Linsolve"}
    ):

        coarsetype, coarseprops = split_off_type(coarseSolve)
        finetype, fineprops = split_off_type(fineSolve)

        self._coarse_solve = globdat[gn.MODULEFACTORY].get_module(coarsetype, "coarse")
        self._fine_solve = globdat[gn.MODULEFACTORY].get_module(finetype, "fine")

        self._coarse_solve.configure(globdat, **coarseprops)
        self._fine_solve.configure(globdat, **fineprops)

    def init(self, globdat):
        pass

    def run(self, globdat):
        self._coarse_solve.solve(globdat["coarse"])
        self._fine_solve.solve(globdat["fine"])

        models = globdat[gn.MODELS]
        fglobdat = globdat["fine"]
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

        # Pass the matrices to the prior model
        for model in self.get_relevant_models("RETURNMATRICES", models):
            model.RETURNMATRICES(globdat)

        model = self.get_unique_relevant_model("GETPRIOR", models)
        prior = model.GETPRIOR(globdat)

        prior = prior.to_direct_gaussian()
        posterior = deepcopy(prior)

        for model in self.get_relevant_models("GETOBSERVATIONOPERATOR", models):
            PhiT = model.GETOBSERVATIONOPERATOR(globdat)
            measurements = model.GETMEASUREMENTS(globdat)

            posterior = posterior.condition_on(PhiT, measurements, 1e-8)

            if isinstance(model, BFEMObservationModel):
                globdat["Phi"] = PhiT.T

            if isinstance(model, BoundaryObservationModel):
                prior = prior.condition_on(PhiT, measurements, 1e-8)

        for model in self.get_relevant_models("APPLYPOSTTRANS", models):
            prior = model.APPLYPOSTTRANS(prior, globdat)
            posterior = model.APPLYPOSTTRANS(posterior, globdat)

        # Create a dictionary for the gp output
        globdat["gp"] = {}
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
        globdat["gp"]["std"]["prior"]["state0"] = np.sqrt(
            np.diagonal(globdat["gp"]["cov"]["prior"]["state0"])
        )
        globdat["gp"]["std"]["prior"]["extForce"] = np.sqrt(
            np.diagonal(globdat["gp"]["cov"]["prior"]["extForce"])
        )
        globdat["gp"]["std"]["posterior"] = {}
        globdat["gp"]["std"]["posterior"]["state0"] = np.sqrt(
            np.diagonal(globdat["gp"]["cov"]["posterior"]["state0"])
        )
        globdat["gp"]["std"]["posterior"]["extForce"] = np.sqrt(
            np.diagonal(globdat["gp"]["cov"]["posterior"]["extForce"])
        )
        globdat["gp"]["samples"] = {}
        globdat["gp"]["samples"]["prior"] = {}
        globdat["gp"]["samples"]["prior"]["state0"] = prior.calc_samples()
        globdat["gp"]["samples"]["prior"]["extForce"] = prior._latent.calc_samples()
        globdat["gp"]["samples"]["posterior"] = {}
        globdat["gp"]["samples"]["posterior"]["state0"] = posterior.calc_samples()
        globdat["gp"]["samples"]["posterior"][
            "extForce"
        ] = posterior._latent.calc_samples()

        return "ok"

    def shutdown(self, globdat):
        pass
