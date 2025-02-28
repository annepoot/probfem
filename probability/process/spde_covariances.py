from warnings import warn
import numpy as np
from scipy.sparse import issparse, diags_array
from scipy.sparse.linalg import spsolve

from copy import deepcopy

from ..multivariate.gaussian import Gaussian
from probability.process.gaussian_process import GaussianProcess
from probability.multivariate import SymbolicCovariance
from .mean_functions import ZeroMeanFunction
from .covariance_functions import CovarianceFunction

from fem.jive import MyJiveRunner
from fem.meshing import create_unit_mass_matrix
from util.linalg import Matrix, MatMulChain

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
        jive_runner=None,
        sigma_pd=1e-8
    ):
        assert isinstance(prior, GaussianProcess)
        assert isinstance(prior.mean, ZeroMeanFunction)
        assert isinstance(
            prior.cov, (InverseCovarianceOperator, NaturalCovarianceOperator)
        )
        self.prior = prior

        if jive_runner is None:
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
            self.module_props = jive_runner.props

        self.props = deepcopy(self.module_props)

        assert "model" not in self.props
        self.props["model"] = self.prior.cov.model_props

        if jive_runner is None:
            self.jive_runner = MyJiveRunner(self.props)
        else:
            self.jive_runner = jive_runner
            self.jive_runner.props = self.props

        self.sigma_pd = sigma_pd
        self.recompute_moments()

    def recompute_moments(self):
        self.globdat = self.jive_runner()
        K = self.globdat["matrix0"]
        self.globdat["matrix0"] = 0.5 * (K + K.T)
        mean = self._compute_mean(self.globdat)
        cov = self._compute_covariance(self.globdat)

        super().__init__(mean, cov, use_scipy_latent=False)

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
            conman = Constrainer(c, K)

            Kc_f = self._constrain_precision(K, c)

            ignore_idx = set([(i, i) for i in c.get_constraints()[0]])
            Kc_f = Matrix(Kc_f, name="Kc", eq_ignore_idx=ignore_idx)
            aKc = MatMulChain(self.prior.cov.scale, Kc_f.inv)

            cov = SymbolicCovariance(aKc)

        elif isinstance(self.prior.cov, NaturalCovarianceOperator):
            K = globdat[gn.MATRIX0]
            c = globdat[gn.CONSTRAINTS]
            conman = Constrainer(c, K)
            Kc = conman.get_output_matrix()

            lumpM = self.prior.cov.lumped_mass_matrix
            Mc_f = self._compute_mass_matrix(globdat, lumped=lumpM)

            ignore_idx = set([(i, i) for i in c.get_constraints()[0]])
            Kc = Matrix(Kc, name="Kc")
            Mc_f = Matrix(Mc_f, name="Mc", eq_ignore_idx=ignore_idx)
            aKMKc = MatMulChain(self.prior.cov.scale, Kc.inv, Mc_f, Kc.inv.T)

            cov = SymbolicCovariance(aKMKc)

        else:
            assert False

        return cov

    def _constrain_covariance(self, matrix, constraints):
        conman = Constrainer(constraints, matrix)
        output_matrix = conman.get_output_matrix()
        cdofs = constraints.get_constraints()[0]
        output_matrix[cdofs, cdofs] = self.sigma_pd**2
        return output_matrix

    def _constrain_precision(self, matrix, constraints):
        conman = Constrainer(constraints, matrix)
        output_matrix = conman.get_output_matrix()
        cdofs = constraints.get_constraints()[0]
        output_matrix[cdofs, cdofs] = self.sigma_pd**-2
        return output_matrix

    def _compute_mass_matrix(self, globdat, lumped):
        elems = globdat[gn.ESET]
        dofs = globdat[gn.DOFSPACE]

        if gn.SHAPE in globdat:
            shape = globdat[gn.SHAPE]
        else:
            node_count = globdat[gn.ESET].max_elem_node_count()
            intscheme = "Gauss" + str(node_count)

            factory = globdat[gn.SHAPEFACTORY]
            shape = factory.get_shape(globdat[gn.MESHSHAPE], intscheme)

        M = create_unit_mass_matrix(elems, dofs, shape, sparse=True, lumped=lumped)
        Mc = self._constrain_covariance(M, globdat[gn.CONSTRAINTS])

        if lumped:
            return diags_array(Mc.diagonal(), format="csr")
        else:
            return Mc
