import numpy as np
from scipy.sparse import issparse, diags, csc_matrix
from scipy.sparse.linalg import inv
import sksparse.cholmod as cm
from warnings import warn

from util.linalg import MatMulChain


__all__ = [
    "Covariance",
]

COVARIANCE = "covariance"
PRECISION = "precision"
DIAGONAL = "diagonal"
SYMBOLIC = "symbolic"


class Covariance:
    """
    This class defines all core functionality for Gaussian distributions.
    All classes that implement some sort of Gaussian are derived from this class
    """

    def __init__(self, A, *, cov_type):
        assert cov_type in [COVARIANCE, PRECISION, DIAGONAL, SYMBOLIC]

        self.cov_type = cov_type
        if cov_type == COVARIANCE:
            assert len(A.shape) == 2
            assert A.shape[0] == A.shape[1]
            self.C = A
            self.shape = A.shape
        elif cov_type == PRECISION:
            assert len(A.shape) == 2
            assert A.shape[0] == A.shape[1]
            self.P = A
            self.shape = A.shape
        elif cov_type == DIAGONAL:
            assert len(A.shape) == 1
            self.D = diags(A)
            self.shape = (len(self.d), len(self.d))
        elif cov_type == SYMBOLIC:
            assert len(A.shape) == 2
            assert A.shape[0] == A.shape[1]
            self.A = A
            self.shape = A.shape
        else:
            assert False

    def __len__(self):
        return self.shape[0]

    def __matmul__(self, other):
        if len(other.shape) in [1, 2]:
            if self.cov_type == COVARIANCE:
                return self.C @ other
            elif self.cov_type == PRECISION:
                chol = self._calc_cholesky()
                if isinstance(chol, cm.Factor):
                    return chol.solve_A(other)
                else:
                    return np.linalg.solve(chol, other)
            elif self.cov_type == DIAGONAL:
                return self.D @ other
            elif self.cov_type == SYMBOLIC:
                return self.A @ other
        else:
            assert False

    def calc_cov(self):
        if self.cov_type == COVARIANCE:
            return self.C
        elif self.cov_type == DIAGONAL:
            return self.D
        elif self.cov_type == PRECISION:
            warn("inverting precision matrix")
            if issparse(self.P):
                return inv(self.P)
            else:
                return np.linalg.inv(self.P)
        elif self.cov_type == SYMBOLIC:
            return self.A.evaluate()
        else:
            assert False

    def calc_cov_inv(self):
        if self.cov_type == COVARIANCE:
            warn("inverting covariance matrix")
            return np.linalg.inv(self.C)
        elif self.cov_type == DIAGONAL:
            return diags(1 / self.D.diagonal())
        elif self.cov_type == PRECISION:
            return self.P
        elif self.cov_type == SYMBOLIC:
            return self.A.inv.evaluate()
        else:
            assert False

    def calc_diag(self):
        return self.calc_cov().diagonal()

    def calc_std(self):
        return np.sqrt(self.calc_diag())

    def calc_sample(self, seed):
        chol = self._calc_cholesky()
        rng = np.random.default_rng(seed)
        std_sample = rng.standard_normal(len(self))

        if self.cov_type == SYMBOLIC:
            assert isinstance(chol, MatMulChain)
            return chol @ std_sample

        else:
            if isinstance(chol, cm.Factor):
                if self.cov_type == COVARIANCE:
                    assert False
                elif self.cov_type == PRECISION:
                    return chol.apply_Pt(
                        chol.solve_Lt(std_sample, use_LDLt_decomposition=False)
                    )
                else:
                    assert False
            else:
                if self.cov_type == PRECISION:
                    return np.linalg.solve(chol.T, std_sample)
                else:
                    return chol @ std_sample

    def calc_samples(self, n, seed):
        chol = self._calc_cholesky()
        rng = np.random.default_rng(seed)
        std_sample = rng.standard_normal((len(self), n))

        if self.cov_type == SYMBOLIC:
            assert isinstance(chol, MatMulChain)
            return (chol @ std_sample).T

        else:
            if isinstance(chol, cm.Factor):
                if self.cov_type == COVARIANCE:
                    assert False
                elif self.cov_type == PRECISION:
                    return chol.apply_Pt(
                        chol.solve_Lt(std_sample, use_LDLt_decomposition=False)
                    ).T
                else:
                    assert False

            else:
                if self.cov_type == PRECISION:
                    return np.linalg.solve(chol.T, std_sample).T
                else:
                    return (chol @ std_sample).T

    def calc_mahal(self, x):
        return np.sqrt(self.calc_mahal_squared(x))

    def calc_mahal_squared(self, x):
        if self.cov_type == COVARIANCE:
            chol = self._calc_cholesky()
            Lx = np.linalg.solve(chol, x)
            return Lx @ Lx
        elif self.cov_type == PRECISION:
            return x @ self.P @ x
        elif self.cov_type == DIAGONAL:
            return x @ (x / self.D.diagonal())
        elif self.cov_type == SYMBOLIC:
            raise NotImplementedError
        else:
            assert False

    def calc_det(self):
        return np.exp(self.calc_logdet())

    def calc_logdet(self):
        chol = self._calc_cholesky()

        if self.cov_type == COVARIANCE:
            return 2 * np.sum(np.log(chol.diagonal()))
        elif self.cov_type == PRECISION:
            if isinstance(chol, cm.Factor):
                return -chol.logdet()
            else:
                return 2 * np.sum(np.log(chol.diagonal()))
        elif self.cov_type == DIAGONAL:
            return 2 * np.sum(np.log(chol.diagonal()))
        elif self.cov_type == SYMBOLIC:
            raise NotImplementedError
        else:
            assert False

    def calc_gram(self, operator):
        if self.cov_type == COVARIANCE:
            return operator @ self.C @ operator.T
        elif self.cov_type == PRECISION:
            chol = self._calc_cholesky()
            sqrtG = chol.solve_L(chol.apply_P(operator.T), use_LDLt_decomposition=False)
            return sqrtG.T @ sqrtG
        elif self.cov_type == DIAGONAL:
            return operator @ self.D @ operator.T
        elif self.cov_type == SYMBOLIC:
            gram_chain = MatMulChain(operator, self.A, operator.T)
            return gram_chain.evaluate()
        else:
            assert False

    def _calc_cholesky(self):
        if not hasattr(self, "_chol"):
            if self.cov_type == COVARIANCE:
                self._chol = np.linalg.cholesky(self.C)
            elif self.cov_type == PRECISION:
                if issparse(self.P):
                    self._chol = cm.cholesky(csc_matrix(self.P))
                else:
                    self._chol = np.linalg.cholesky(self.P)
            elif self.cov_type == DIAGONAL:
                self._chol = np.sqrt(self.D)
            elif self.cov_type == SYMBOLIC:
                self._chol = self.A.factorize()
            else:
                assert False
        return self._chol
