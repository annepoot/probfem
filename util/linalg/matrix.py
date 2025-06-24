import warnings
import numpy as np
from scipy.sparse import issparse, eye_array, diags_array, csc_matrix
from sksparse import cholmod as cm
from copy import copy

__all__ = ["Matrix", "MatMulChain"]


class Matrix:

    tol = 1e-12

    def __init__(
        self,
        matrix,
        *,
        name=None,
        eq_ignore_idx=set(),
        invert=None,
        transpose=None,
    ):
        if isinstance(matrix, Matrix):
            self.matrix = matrix.matrix
            self.name = matrix.name

            self.eq_ignore_idx = matrix.eq_ignore_idx

            if invert is None:
                self.is_inverted = matrix.is_inverted
            else:
                self.is_inverted = invert

            if transpose is None:
                self.is_transposed = matrix.is_transposed
            else:
                self.is_transposed = transpose

            self._is_symmetric = matrix._is_symmetric
            self._is_diagonal = matrix._is_diagonal

        else:
            if issparse(matrix) or isinstance(matrix, np.ndarray):
                assert len(matrix.shape) == 2
            elif isinstance(matrix, cm.Factor):
                pass
            else:
                assert False

            self.matrix = matrix
            self.name = name
            self.eq_ignore_idx = eq_ignore_idx

            if invert is None:
                self.is_inverted = False
            else:
                assert isinstance(invert, bool)
                self.is_inverted = invert

            if transpose is None:
                self.is_transposed = False
            else:
                assert isinstance(transpose, bool)
                self.is_transposed = transpose

            self._is_symmetric = None
            self._is_diagonal = None

        if self.is_symmetric:
            self.is_transposed = False

    def __repr__(self):
        if self.is_inverted:
            if self.is_transposed:
                return f"{self.name}^-T"
            else:
                return f"{self.name}^-1"
        else:
            if self.is_transposed:
                return f"{self.name}^T"
            else:
                return f"{self.name}"

    def __eq__(self, other):
        return self.is_equal(other, tol=0)

    def __mul__(self, other):
        assert np.isscalar(other)
        return MatMulChain(other, self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        assert np.isscalar(other)
        return self * (1 / other)

    def __matmul__(self, other):
        if isinstance(other, (Matrix, MatMulChain)):
            return MatMulChain(self, other)
        else:
            assert issparse(other) or isinstance(other, np.ndarray)
            assert self.shape[1] == other.shape[0]
            assert len(other.shape) in [1, 2]

            if self.is_inverted:
                if self.is_diagonal:
                    return self.evaluate() @ other
                elif self.is_factor:
                    if issparse(other):
                        rhs = other.toarray()
                    else:
                        rhs = other

                    if self.is_transposed:
                        sol = self.matrix.solve_Lt(rhs, use_LDLt_decomposition=False)
                        sol = self.matrix.apply_Pt(sol)
                        return sol
                    else:
                        rhs = self.matrix.apply_P(rhs)
                        sol = self.matrix.solve_L(rhs, use_LDLt_decomposition=False)
                        return sol

                elif len(other.shape) == 1 or other.shape[0] > other.shape[1]:
                    if self.is_sparse:
                        array = self.inv.evaluate()
                        factor = cm.cholesky(csc_matrix(array))
                        if issparse(other):
                            return factor.solve_A(other.toarray())
                        else:
                            return factor.solve_A(other)
                    else:
                        array = self.inv.evaluate()
                        if issparse(other):
                            return np.linalg.solve(array, other.toarray())
                        else:
                            return np.linalg.solve(array, other)
                else:
                    return self.evaluate() @ other

            else:
                return self.evaluate() @ other

    def __rmatmul__(self, other):
        return (self.T @ other.T).T

    @property
    def shape(self):
        if self.is_factor:
            n = len(self.matrix.P())
            return (n, n)
        else:
            if self.is_transposed:
                return self.matrix.shape[::-1]
            else:
                return self.matrix.shape

    @property
    def is_sparse(self):
        return issparse(self.matrix)

    @property
    def is_diagonal(self):
        if self._is_diagonal is None:
            if self.is_factor:
                self._is_diagonal = False
            else:
                if self.is_sparse:
                    if self.shape[0] == self.shape[1]:
                        if np.count_nonzero(self.matrix.diagonal()) == self.matrix.nnz:
                            self._is_diagonal = True
                        else:
                            self._is_diagonal = False
                    else:
                        self._is_diagonal = False
                else:
                    self._is_diagonal = False

        return self._is_diagonal

    @property
    def is_symmetric(self):
        if self._is_symmetric is None:
            if self.is_factor:
                self._is_symmetric = False
            elif self.is_diagonal:
                self._is_symmetric = True
            else:
                if self.shape[0] == self.shape[1]:
                    if self.is_sparse:
                        diff = abs(self.matrix - self.matrix.T)
                        check = diff - abs(self.matrix) * self.tol > self.tol
                        self._is_symmetric = check.nnz == 0
                    else:
                        diff = abs(self.matrix - self.matrix.T)
                        check = diff > abs(self.matrix) * self.tol + self.tol
                        self._is_symmetric = not np.any(check)
                else:
                    self._is_symmetric = False

        return self._is_symmetric

    @property
    def is_factor(self):
        return isinstance(self.matrix, cm.Factor)

    @property
    def T(self):
        if self.is_symmetric:
            return self
        else:
            return Matrix(self, transpose=not self.is_transposed)

    @property
    def inv(self):
        return Matrix(self, invert=not self.is_inverted)

    def evaluate(self):
        if self.is_inverted:
            assert not self.is_factor

            if self.is_diagonal:
                diag = self.matrix.diagonal()
                inv = diags_array(1 / diag)
                return inv

            else:
                warnings.warn("explicit matrix inversion")
                if self.is_sparse:
                    not_inv = self.matrix.todense()
                else:
                    not_inv = self.matrix

                if self.is_transposed:
                    return np.linalg.inv(not_inv.T)
                else:
                    return np.linalg.inv(not_inv)

        else:
            if self.is_factor:
                with warnings.catch_warnings():
                    msg = "array contains 32 bit integers; but 64 bit integers are needed; slowing down due to converting"
                    warnings.filterwarnings("ignore", message=msg)
                    array = self.matrix.apply_Pt(self.matrix.L())
            else:
                array = self.matrix

            if self.is_transposed:
                return array.T
            else:
                return array

    def factorize(self):
        assert self.is_symmetric

        if not hasattr(self, "_factor"):
            if self.is_sparse:
                if self.is_diagonal:
                    factor = diags_array(np.sqrt(self.matrix.diagonal()))
                else:
                    factor = cm.cholesky(csc_matrix(self.matrix))
            else:
                factor = np.linalg.cholesky(self.matrix)

            sqrt_name = f"sqrt({self.name})"
            if self.is_inverted:
                self._factor = Matrix(
                    factor, invert=True, transpose=True, name=sqrt_name
                )
            else:
                self._factor = Matrix(
                    factor, invert=False, transpose=False, name=sqrt_name
                )

        return self._factor

    def calc_logdet(self):
        chol = self.factorize()

        if self.is_inverted:
            if chol.is_factor:
                return -chol.matrix.logdet()
            else:
                d = chol.inv.evaluate().diagonal()
                return -2 * np.sum(np.log(d))
        else:
            if chol.is_factor:
                return chol.matrix.logdet()
            else:
                d = chol.evaluate().diagonal()
                return 2 * np.sum(np.log(d))

    def is_equal(self, other, tol=0):
        if self.is_factor != other.is_factor:
            return False
        elif self.is_sparse != other.is_sparse:
            return False
        elif self.is_inverted != other.is_inverted:
            return False
        elif self.is_symmetric != other.is_symmetric:
            return False
        elif self.is_transposed != other.is_transposed:
            return self.is_equal(other.T, tol=tol)
        elif self.shape != other.shape:
            return False
        else:
            if issparse(self.matrix):
                diff = self.matrix != other.matrix
                if diff.nnz == 0:
                    return True

                eq_ignore_idx = self.eq_ignore_idx.union(other.eq_ignore_idx)
                if diff.nnz > len(eq_ignore_idx):
                    return False
                else:
                    for row, col in zip(*diff.nonzero()):
                        if (row, col) not in eq_ignore_idx:
                            return False
                    return True
            else:
                return np.all(self.matrix == other.matrix)


class MatMulChain:

    def __init__(self, *entries, skip_check=False):
        self.chain = []
        self.scale = 1.0

        for entry in entries:
            self.append(entry, skip_check=skip_check)

    def __repr__(self):
        return " ".join([repr(matrix) for matrix in self])

    def __len__(self):
        return len(self.chain)

    def __getitem__(self, idx):
        return self.chain[idx]

    def __iter__(self):
        return iter(self.chain)

    def __next__(self):
        return next(self.chain)

    def __copy__(self):
        new = MatMulChain(self.scale, *self.chain, skip_check=True)
        return new

    def __mul__(self, other):
        assert np.isscalar(other)
        new = copy(self)
        new.scale *= other
        return new

    def __rmul__(self, other):
        return self * other

    def __imul__(self, other):
        assert np.isscalar(other)
        self.scale *= other
        return self

    def __truediv__(self, other):
        assert np.isscalar(other)
        return self * (1 / other)

    def __matmul__(self, other):
        result = other
        for matrix in self[::-1]:
            result = matrix @ result
        result *= self.scale
        return result

    def __rmatmul__(self, other):
        result = other
        for matrix in self:
            result = result @ matrix
        result *= self.scale
        return result

    @property
    def shape(self):
        return (self[0].shape[0], self[-1].shape[1])

    @property
    def T(self):
        lst = [matrix.T for matrix in self[::-1]]
        return MatMulChain(self.scale, *lst, skip_check=True)

    @property
    def inv(self):
        lst = [matrix.inv for matrix in self[::-1]]
        return MatMulChain(1 / self.scale, *lst, skip_check=True)

    @property
    def is_symmetric(self):
        n = len(self)

        for i in range(n // 2):
            if self[i] != self[n - i - 1].T:
                return False
        if n % 2 == 0:
            return True
        else:
            return self[(n - 1) // 2].is_symmetric

    @property
    def is_diagonal(self):
        for mat in self:
            if not mat.is_diagonal:
                return False
        return True

    def append(self, item, *, skip_check=False):
        if np.isscalar(item):
            self.scale *= item
        elif isinstance(item, MatMulChain):
            self.scale *= item.scale
            for entry in item:
                self.append(entry)
        else:
            entry = Matrix(item)
            self.chain.append(entry)

        if not skip_check:
            self.check_chain()

    def prepend(self, item, *, skip_check=False):
        if np.isscalar(item):
            self.scale *= item
        elif isinstance(item, MatMulChain):
            self.scale *= item.scale
            for entry in item[::-1]:
                self.prepend(entry)
        else:
            entry = Matrix(item)
            self.chain.insert(0, entry)

        if not skip_check:
            self.check_chain()

    def simplify(self, tol=0):
        # Check for patterns like K K^-1 K (keep middle K)
        for i in range(len(self) - 1, 1, -1):
            entry1, entry2, entry3 = self[i - 2], self[i - 1], self[i]
            if entry1.is_equal(entry2.inv) and entry3.is_equal(entry2.inv):
                self.chain.pop(i)
                self.chain[i - 1] = self.chain[i - 1].inv
                self.chain.pop(i - 2)
                self.simplify()
                return self

        # Check for patterns like K^-1 K (replace with identity)
        for i in range(len(self) - 1, 0, -1):
            entry1, entry2 = self[i - 1], self[i]
            if entry1.is_equal(entry2.inv):
                self.chain.pop(i)
                self.chain.pop(i - 1)
                self.simplify()
                return self

        return self

    def evaluate(self):
        self.simplify()

        lst = [mat for mat in self]

        n = len(lst)
        mid = (n - 1) // 2
        use_symmetry = n > 1 and self.is_symmetric and lst[mid].is_inverted

        if n > 1:
            if use_symmetry:
                lst = lst[: mid + 1]

                if n % 2 == 1:
                    lst[mid] = lst[mid].factorize()

            for _ in range(100):
                cost = np.zeros(len(lst) - 1)
                for i in range(len(lst) - 1):
                    mati = lst[i]
                    matj = lst[i + 1]

                    if isinstance(mati, Matrix) and isinstance(matj, Matrix):
                        if mati.is_inverted and matj.is_inverted:
                            cost[i] = np.inf
                        else:
                            cost[i] = mati.shape[0] * mati.shape[1] * matj.shape[1]
                    else:
                        cost[i] = mati.shape[0] * mati.shape[1] * matj.shape[1]

                opt_idx = np.argmin(cost)
                mati = lst[opt_idx]
                matj = lst[opt_idx + 1]

                if isinstance(mati, Matrix):
                    if not mati.is_inverted:
                        mati = mati.evaluate()

                if isinstance(matj, Matrix):
                    if not matj.is_inverted:
                        matj = matj.evaluate()

                if isinstance(mati, np.ndarray) and isinstance(matj, Matrix):
                    lst[opt_idx] = (matj.T @ mati.T).T
                else:
                    lst[opt_idx] = mati @ matj
                lst.pop(opt_idx + 1)

                if len(lst) == 1:
                    break

        result = lst[0]

        if isinstance(result, Matrix):
            result = result.evaluate()

        if issparse(result):
            if result.nnz > 0.5 * np.prod(result.shape):
                result = result.toarray()

        if use_symmetry:
            result = result @ result.T

        result *= self.scale

        return result

    def factorize(self):
        assert self.is_symmetric

        self.simplify()

        n = len(self)
        if n % 2 == 0:
            mid = n // 2
            return MatMulChain(np.sqrt(self.scale), *self[:mid], skip_check=True)
        else:
            mid = (n - 1) // 2
            factor = self[mid].factorize()
            return MatMulChain(
                np.sqrt(self.scale), *self[:mid], factor, skip_check=True
            )

    def check_chain(self):
        for i in range(len(self)):
            shape = self[i].shape
            if len(shape) != 2:
                raise ValueError

            if i >= 1:
                if shape[0] != self[i - 1].shape[1]:
                    raise ValueError
