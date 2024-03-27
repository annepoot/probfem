import numpy as np

from myjive.names import GlobNames as gn
from myjive.app import Module
from myjive.util.proputils import mdtarg, mdtdict, optarg
from myjive.util import Table
from copy import deepcopy


class RMFemModule(Module):
    def init(self, globdat, **props):
        # Get props
        solvemoduleprops = mdtdict(self, props, "solveModule", ["type"])
        modelprops = mdtdict(self, props, "modelprops", mandatory_keys=["models"])
        self._nsample = mdtarg(self, props, "nsample", dtype=int)
        writemeshprops = optarg(self, props, "writeMesh", dtype=dict)
        seed = optarg(self, props, "seed")
        self._rng = np.random.default_rng(seed)

        if "file" in writemeshprops and "type" in writemeshprops:
            self._writemeshfile = writemeshprops["file"]
            self._writemeshtype = writemeshprops["type"]
        else:
            self._writemeshfile = None

        modulefac = globdat[gn.MODULEFACTORY]
        solvemoduletype = solvemoduleprops["type"]
        self._solvemodule = modulefac.get_module(solvemoduletype, "solveModule")
        self._solvemodule.init(globdat, **solvemoduleprops)

        perturbed_solves = []
        for _ in range(self._nsample):
            pglobdat = deepcopy(globdat)
            modelfac = pglobdat[gn.MODELFACTORY]

            name_list = modelprops["models"]
            model_list = []
            for name in name_list:
                m = modelfac.get_model(modelprops[name]["type"], name)
                m.configure(pglobdat, **(modelprops[name]))
                model_list.append(m)
            pglobdat[gn.MODELS] = model_list

            perturbed_solves.append(pglobdat)

        globdat["perturbedSolves"] = perturbed_solves

    def run(self, globdat):
        # Perform unperturbed solve
        self._solvemodule.solve(globdat)

        # Get element size table
        meshsize = Table(size=len(globdat[gn.ESET]))
        for model in self.get_relevant_models("GETELEMTABLE", globdat[gn.MODELS]):
            meshsize = model.GETELEMTABLE("size", meshsize, globdat)

        for i, pglobdat in enumerate(globdat["perturbedSolves"]):
            nodes = pglobdat[gn.NSET]
            models = pglobdat[gn.MODELS]

            for model in self.get_relevant_models("PERTURBNODES", models):
                nodes = model.PERTURBNODES(nodes, pglobdat, meshsize=meshsize, rng=self._rng)

            # Perform the solve
            self._solvemodule.solve(pglobdat)

            # Write the mesh to a file
            if self._writemeshfile is not None:
                for model in self.get_relevant_models("WRITEMESH", models):
                    fname, ftype = self._writemeshfile.format(i), self._writemeshtype
                    model.WRITEMESH(pglobdat, fname=fname, ftype=ftype)

        return "ok"

    def shutdown(self, globdat):
        pass
