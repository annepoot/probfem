from myjive.names import GlobNames as gn

from myjive.app import ModuleFactory
from myjive.model import ModelFactory

from .bfemmodel import BFEMModel
from .bfemobservationmodel import BFEMObservationModel
from .boundaryobservationmodel import BoundaryObservationModel
from .cgobservationmodel import CGObservationModel
from .randomobservationmodel import RandomObservationModel

from .bfemsolvemodule import BFEMSolveModule
from .conversionmodule import ConversionModule
from .probviewmodule import ProbViewModule

__all__ = ["declare_all", "declare_models", "declare_modules"]


def declare_all(globdat):
    declare_models(globdat)
    declare_modules(globdat)


def declare_models(globdat):
    factory = globdat.get(gn.MODELFACTORY, ModelFactory())

    BFEMModel.declare(factory)
    BFEMObservationModel.declare(factory)
    BoundaryObservationModel.declare(factory)
    CGObservationModel.declare(factory)
    RandomObservationModel.declare(factory)

    globdat[gn.MODELFACTORY] = factory


def declare_modules(globdat):
    factory = globdat.get(gn.MODULEFACTORY, ModuleFactory())

    BFEMSolveModule.declare(factory)
    ConversionModule.declare(factory)
    ProbViewModule.declare(factory)

    globdat[gn.MODULEFACTORY] = factory
