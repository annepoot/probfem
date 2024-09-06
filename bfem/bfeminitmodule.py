from copy import deepcopy

from myjive.names import GlobNames as gn
from myjive.app import Module, InitModule
from myjive.util.proputils import split_off_type


class BFEMInitModule(Module):
    def __init__(self, name):
        super().__init__(name)
        self._needs_modelprops = True

    @InitModule.save_config
    def configure(
        self,
        globdat,
        *,
        coarseInit={"type": "Init"},
        fineInit={"type": "Init"},
        mesh,
        nodeGroups=[],
        elemGroups=[],
        **groupprops
    ):

        sharedprops = {"mesh": mesh, "nodeGroups": nodeGroups, "elemGroups": elemGroups}
        sharedprops.update(groupprops)

        coarseprops = deepcopy(sharedprops)
        fineprops = deepcopy(sharedprops)

        coarsetype, cprops = split_off_type(coarseInit)
        finetype, fprops = split_off_type(fineInit)

        coarseprops.update(cprops)
        fineprops.update(fprops)

        cglobdat = deepcopy(globdat)
        fglobdat = deepcopy(globdat)

        globdat["coarse"] = cglobdat
        globdat["fine"] = fglobdat

        self._coarse_init = cglobdat[gn.MODULEFACTORY].get_module(coarsetype, "coarse")
        self._fine_init = fglobdat[gn.MODULEFACTORY].get_module(finetype, "fine")

        self._coarse_init.configure(cglobdat, **coarseprops)
        self._fine_init.configure(fglobdat, **fineprops)

    def init(self, globdat, *, modelprops):
        self._coarse_init.init(globdat["coarse"], modelprops=modelprops)
        self._fine_init.init(globdat["fine"], modelprops=modelprops)

        globdat[gn.MODELS] = deepcopy(globdat["fine"][gn.MODELS])

    def run(self, globdat):
        return "ok"

    def shutdown(self, globdat):
        pass
