from myjive.names import GlobNames as gn
from myjive.app import Module
from myjive.util.proputils import mandatory_dict, mandatory_argument, optional_argument
from copy import deepcopy


class RMFemModule(Module):

    def init(self, globdat, **props):

        # Get props
        solvemoduleprops = mandatory_dict(
            self, props, "solveModule", mandatory_keys=["type"]
        )
        self._nsample = mandatory_argument(self, props, "nsample")

        modulefac = globdat[gn.MODULEFACTORY]
        solvemoduletype = solvemoduleprops["type"]
        self._solvemodule = modulefac.get_module(solvemoduletype, "solveModule")
        self._solvemodule.init(globdat, **solvemoduleprops)

        self._models = globdat[gn.MODELS]

    def run(self, globdat):

        # Perform unperturbed solve
        self._solvemodule.solve(globdat)

        # Generate perturbed meshes
        perturbed_solves = []

        for i in range(self._nsample):
            pglobdat = deepcopy(globdat)
            nodes = pglobdat[gn.NSET]

            for model in self.get_relevant_models("PERTURBNODES", self._models):
                nodes = model.PERTURBNODES(nodes, pglobdat)

            # Perform the solve
            self._solvemodule.solve(pglobdat)

            perturbed_solves.append(pglobdat)

        # Store the perturbed solutions in globdat
        globdat["perturbedSolves"] = perturbed_solves

        return "ok"

    def shutdown(self, globdat):
        pass
