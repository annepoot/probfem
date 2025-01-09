from copy import deepcopy
from warnings import warn
import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import spsolve
from scipy.stats import Covariance

from ..multivariate.gaussian import Gaussian
from probability.process.gaussian_process import GaussianProcess
from .mean_functions import ZeroMeanFunction
from .covariance_functions import CovarianceFunction

from fem.jive import MyJiveRunner
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

    def __init__(
        self,
        prior,
        init_props=None,
        solve_props=None,
        *,
        module_props=None,
        jive_runner=None,
        jive_runner_kws={},
        sigma_pd=1e-8
    ):
        assert isinstance(prior, GaussianProcess)
        assert isinstance(prior.mean, ZeroMeanFunction)
        assert isinstance(
            prior.cov, (InverseCovarianceOperator, NaturalCovarianceOperator)
        )
        self.prior = prior

        if module_props is None:
            assert init_props is not None
            assert solve_props is not None
            warn(
                "using init_props + solve_props, but should really use module_props instead"
            )
            self.module_props = {
                "modules": ["init", "solve"],
                "init": init_props,
                "solve": solve_props,
            }
        else:
            self.module_props = module_props

        self.props = deepcopy(self.module_props)

        assert "model" not in self.props
        self.props["model"] = self.prior.cov.model_props

        self.sigma_pd = sigma_pd

        if jive_runner is None:
            jive = MyJiveRunner(self.props)
        else:
            jive = jive_runner(self.props, **jive_runner_kws)

        self.globdat = jive()

        mean = self._compute_mean(self.globdat)
        cov = self._compute_covariance(self.globdat)

        super().__init__(mean, cov)

    def update_mean(self):
        assert False

    def update_cov(self):
        assert False

    def _compute_mean(self, globdat):
        K = globdat[gn.MATRIX0].copy()
        c = globdat[gn.CONSTRAINTS]

        cdofs, cvals = c.get_constraints()
        cdofs = np.array(cdofs)
        cvals = np.array(cvals)
        idx_inhom = np.where(abs(cvals) > 1e-8)

        if len(idx_inhom[0]) == 0:
            mean = np.zeros(K.shape[0])

        else:
            Kib = K[:, cdofs]
            f_inhom = -Kib @ cvals
            f_inhom[cdofs] = cvals

            K[:, cdofs] *= 0.0
            K[cdofs, :] *= 0.0
            K[cdofs, cdofs] = 1.0

            if issparse(K):
                mean = spsolve(K, f_inhom)
            else:
                mean = np.linalg.solve(K, f_inhom)

        return mean

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
