from myjive.names import GlobNames as gn
from myjive.app import Module
from myjive.util.proputils import mdtarg, mdtdict, optarg
from copy import deepcopy


class RMFemModule(Module):
    def init(self, globdat, **props):
        # Get props
        solvemoduleprops = mdtdict(self, props, "solveModule", ["type"])
        self._nsample = mdtarg(self, props, "nsample", dtype=int)
        writemeshprops = optarg(self, props, "writeMesh", dtype=dict)

        if "file" in writemeshprops and "type" in writemeshprops:
            self._writemeshfile = writemeshprops["file"]
            self._writemeshtype = writemeshprops["type"]
        else:
            self._writemeshfile = None

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

            # Write the mesh to a file
            if self._writemeshfile is not None:
                for model in self.get_relevant_models("WRITEMESH", self._models):
                    fname, ftype = self._writemeshfile.format(i), self._writemeshtype
                    model.WRITEMESH(pglobdat, fname=fname, ftype=ftype)

        # Store the perturbed solutions in globdat
        globdat["perturbedSolves"] = perturbed_solves

        return "ok"

    def shutdown(self, globdat):
        pass
