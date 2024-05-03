from myjive.names import GlobNames as gn

from myjive.app import ModuleFactory
from myjive.model import ModelFactory

from observationmodel import ObservationModel
from randommeshmodel import RandomMeshModel
from referencemodel import ReferenceModel

from mcmcmodule import MCMCModule
from rmfemmodule import RMFemModule
from rmplotmodule import RMPlotModule


def declare_all(globdat):
    declare_extra_models(globdat)
    declare_extra_modules(globdat)


def declare_extra_models(globdat):
    factory = globdat.get(gn.MODELFACTORY, ModelFactory())

    ObservationModel.declare(factory)
    RandomMeshModel.declare(factory)
    ReferenceModel.declare(factory)

    globdat[gn.MODELFACTORY] = factory


def declare_extra_modules(globdat):
    factory = globdat.get(gn.MODULEFACTORY, ModuleFactory())

    MCMCModule.declare(factory)
    RMFemModule.declare(factory)
    RMPlotModule.declare(factory)

    globdat[gn.MODULEFACTORY] = factory
