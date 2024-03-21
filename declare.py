from myjive.names import GlobNames as gn

from myjive.app import ModuleFactory
from myjive.model import ModelFactory

from randommeshmodel import RandomMeshModel

from rmfemmodule import RMFemModule


def declare_all(globdat):
    declare_extra_models(globdat)
    declare_extra_modules(globdat)


def declare_extra_models(globdat):
    factory = globdat.get(gn.MODELFACTORY, ModelFactory())

    RandomMeshModel.declare(factory)

    globdat[gn.MODELFACTORY] = factory


def declare_extra_modules(globdat):
    factory = globdat.get(gn.MODULEFACTORY, ModuleFactory())

    RMFemModule.declare(factory)

    globdat[gn.MODULEFACTORY] = factory
