from myjive.names import GlobNames as gn

from myjive.app import ModuleFactory
from myjive.model import ModelFactory

from .randommeshmodel import RandomMeshModel
from .referencemodel import ReferenceModel

from .rmfemmodule import RMFemModule
from .rmplotmodule import RMPlotModule

__all__ = ["declare_all", "declare_models", "declare_modules"]


def declare_all(globdat):
    declare_models(globdat)
    declare_modules(globdat)


def declare_models(globdat):
    factory = globdat.get(gn.MODELFACTORY, ModelFactory())

    RandomMeshModel.declare(factory)
    ReferenceModel.declare(factory)

    globdat[gn.MODELFACTORY] = factory


def declare_modules(globdat):
    factory = globdat.get(gn.MODULEFACTORY, ModuleFactory())

    RMFemModule.declare(factory)
    RMPlotModule.declare(factory)

    globdat[gn.MODULEFACTORY] = factory
