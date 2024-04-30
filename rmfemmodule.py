import numpy as np

from myjive.names import GlobNames as gn
from myjive.app import Module
from myjive.util.proputils import check_dict, split_off_type
from myjive.util import Table
from copy import deepcopy


class RMFemModule(Module):

    def __init__(self, name):
        super().__init__(name)
        self._needs_modelprops = True

    def configure(
        self,
        globdat,
        *,
        solveModule,
        nsample,
        writeMesh={},
        seed=None,
        errorTables=[],
        estimatorTables=[]
    ):
        # Validate input arguments
        check_dict(self, solveModule, ["type"])
        if len(writeMesh) > 0:
            check_dict(self, writeMesh, ["file", "type"])
        self._nsample = nsample
        self._rng = np.random.default_rng(seed)
        self._errornames = errorTables
        self._estimatornames = estimatorTables

        if "file" in writeMesh and "type" in writeMesh:
            self._writemeshfile = writeMesh["file"]
            self._writemeshtype = writeMesh["type"]
        else:
            self._writemeshfile = None

        modulefac = globdat[gn.MODULEFACTORY]
        solvetype, solveprops = split_off_type(solveModule)
        self._solvemodule = modulefac.get_module(solvetype, "solveModule")
        self._solvemodule.configure(globdat, **solveprops)

    def init(self, globdat, *, modelprops):
        # Validate input arguments
        check_dict(self, modelprops, ["models"])

        # Get props
        perturbed_solves = []
        for _ in range(self._nsample):
            pglobdat = deepcopy(globdat)
            modelfac = pglobdat[gn.MODELFACTORY]

            name_list = modelprops["models"]
            model_list = []
            for name in name_list:
                typ, mprops = split_off_type(modelprops[name])
                m = modelfac.get_model(typ, name)
                m.configure(pglobdat, **mprops)
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
                nodes = model.PERTURBNODES(
                    nodes, pglobdat, meshsize=meshsize, rng=self._rng
                )

            # Perform the solve
            self._solvemodule.solve(pglobdat)

            # Write the mesh to a file
            if self._writemeshfile is not None:
                for model in self.get_relevant_models("WRITEMESH", models):
                    fname, ftype = self._writemeshfile.format(i), self._writemeshtype
                    model.WRITEMESH(pglobdat, fname=fname, ftype=ftype)

        if gn.TABLES not in globdat:
            globdat[gn.TABLES] = {}

        errortable = Table(size=len(globdat[gn.ESET]))
        for name in self._errornames:
            for model in self.get_relevant_models("COMPUTEERROR", globdat[gn.MODELS]):
                errortable = model.COMPUTEERROR(name, errortable, globdat)

        for name in self._estimatornames:
            for model in self.get_relevant_models(
                "COMPUTEESTIMATOR", globdat[gn.MODELS]
            ):
                errortable = model.COMPUTEESTIMATOR(name, errortable, globdat)

        globdat[gn.TABLES]["error"] = errortable

        return "ok"

    def shutdown(self, globdat):
        pass
