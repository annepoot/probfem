import numpy as np

from myjive.names import GlobNames as gn
from myjive.model.model import Model
from myjive.util.proputils import split_off_type


class BFEMReferenceModel(Model):
    def RETURNMATRICES(self, globdat):
        self._init.run(self._refdat)
        self._solver.run(self._refdat)
        globdat["ref"][self.get_name()] = self._refdat

    def GETREFDAT(self, globdat):
        return self._refdat

    @Model.save_config
    def configure(self, globdat, init, models, solver):
        inittype, newinitprops = split_off_type(init)
        solvertype, newsolverprops = split_off_type(solver)
        self._init = globdat[gn.MODULEFACTORY].get_module(inittype, "refinit")
        self._solver = globdat[gn.MODULEFACTORY].get_module(solvertype, "refsolve")

        initprops = {}
        solverprops = {}
        for module in globdat[gn.MODULES].values():
            if isinstance(module, type(self._init)):
                if len(initprops) > 0:
                    raise ValueError("ambiguous init props")
                initprops.update(module.get_config())
            elif isinstance(module, type(self._solver)):
                if len(solverprops) > 0:
                    raise ValueError("ambiguous solver props")
                solverprops.update(module.get_config())

        initprops.update(newinitprops)
        solverprops.update(newsolverprops)

        self._refdat = {
            gn.MODULEFACTORY: globdat[gn.MODULEFACTORY],
            gn.MODELFACTORY: globdat[gn.MODELFACTORY],
            gn.SHAPEFACTORY: globdat[gn.SHAPEFACTORY],
            gn.SOLVERFACTORY: globdat[gn.SOLVERFACTORY],
            gn.PRECONFACTORY: globdat[gn.PRECONFACTORY],
        }

        self._init.configure(self._refdat, **initprops)
        self._solver.configure(self._refdat, **solverprops)

        model_list = []
        modelprops = {}
        modelprops["models"] = models
        for model_name in models:
            for m in globdat[gn.MODELS].values():
                if m.get_name() == model_name:
                    model_list.append(m)
                    modelprops[model_name] = m.get_config()
                    break
            else:
                raise ValueError("Model '{}' not found!".format(model_name))

        self._init.init(self._refdat, modelprops=modelprops)
        self._solver.init(self._refdat)
