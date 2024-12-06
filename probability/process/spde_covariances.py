import numpy as np
from scipy.sparse import issparse
from scipy.stats import Covariance

from ..multivariate.gaussian import Gaussian
from probability.process.gaussian_process import GaussianProcess
from .mean_functions import ZeroMeanFunction
from .covariance_functions import CovarianceFunction

from fem.jive import JiveRunner
from fem.meshing import create_unit_mass_matrix

from myjive.names import GlobNames as gn
from myjive.solver import Constrainer

__all__ = [
    "CovarianceOperator",
    "InverseCovarianceOperator",
    "NaturalCovarianceOperator",
    "ProjectedPrior",
]


class CovarianceOperator(CovarianceFunction):
    def calc_cov(self, v1, v2):
        raise NotImplementedError("This has to be implemented in a child class")


class InverseCovarianceOperator(CovarianceOperator):
    def __init__(self, *, model_props, scale):
        self.model_props = model_props
        self.scale = scale

    def calc_cov(self, v1, v2):
        # covariance implied by diffop^-1
        raise NotImplementedError(
            "This covariance operator cannot be evaluated analytically"
        )


class NaturalCovarianceOperator(CovarianceOperator):
    def __init__(self, *, model_props, scale, lumped_mass_matrix):
        self.model_props = model_props
        self.scale = scale
        self.lumped_mass_matrix = lumped_mass_matrix

    def calc_cov(self, v1, v2):
        # covariance implied by diffop^-2
        raise NotImplementedError(
            "This covariance operator cannot be evaluated analytically"
        )


class ProjectedPrior(Gaussian):

    def __init__(self, *, prior, init_props, solve_props, sigma_pd=1e-8):
        assert isinstance(prior, GaussianProcess)
        assert isinstance(prior.mean, ZeroMeanFunction)
        assert isinstance(
            prior.cov, (InverseCovarianceOperator, NaturalCovarianceOperator)
        )
        self.prior = prior

        self.props = {
            "modules": ["init", "solve"],
            "init": init_props,
            "solve": solve_props,
            "model": self.prior.cov.model_props,
        }

        self.sigma_pd = sigma_pd

        jive = JiveRunner(self.props)
        self.globdat = jive()

        cov = self._compute_covariance(self.globdat)

        super().__init__(None, cov)

    def update_mean(self):
        assert False

    def update_cov(self):
        assert False

    def _compute_covariance(self, globdat):

        if isinstance(self.prior.cov, InverseCovarianceOperator):
            K = globdat[gn.MATRIX0]
            c = globdat[gn.CONSTRAINTS]
            if issparse(K):
                K = K.toarray()

            scale = self.prior.cov.scale
            aK = K / scale

            aKc = self._constrain_precision(aK, c)
            cov = Covariance.from_precision(aKc)

        elif isinstance(self.prior.cov, NaturalCovarianceOperator):
            K = globdat[gn.MATRIX0]
            if issparse(K):
                K = K.toarray()

            scale = self.prior.cov.scale
            lumpM = self.prior.cov.lumped_mass_matrix
            M_inv = self._compute_inv_mass_matrix(globdat, lumped=lumpM)
            aKMK = (K @ M_inv @ K) / scale

            aKMKc = self._constrain_precision(aKMK, globdat[gn.CONSTRAINTS])
            cov = Covariance.from_precision(aKMKc)

        else:
            assert False

        return cov

    def _constrain_precision(self, matrix, constraints):
        conman = Constrainer(constraints, matrix)
        output_matrix = conman.get_output_matrix()
        cdofs = constraints.get_constraints()[0]
        output_matrix[cdofs, cdofs] = self.sigma_pd**-2
        if issparse(output_matrix):
            output_matrix = output_matrix.toarray()
        return output_matrix

    def _compute_inv_mass_matrix(self, globdat, lumped):
        elems = globdat[gn.ESET]
        dofs = globdat[gn.DOFSPACE]
        node_count = globdat[gn.ESET].max_elem_node_count()
        intscheme = "Gauss" + str(node_count)
        shape = globdat[gn.SHAPEFACTORY].get_shape(globdat[gn.MESHSHAPE], intscheme)

        M = create_unit_mass_matrix(elems, dofs, shape, sparse=True, lumped=lumped)
        Mc = Constrainer(globdat[gn.CONSTRAINTS], M).get_output_matrix()

        if lumped:
            M_inv = np.diag(1 / Mc.diagonal())
        else:
            M_inv = np.linalg.inv(Mc.toarray())

        return M_inv
