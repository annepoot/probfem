import warnings
import numpy as np
from scipy.sparse import issparse, eye_array, diags, csc_matrix
from scipy.sparse.linalg import spsolve
from sksparse import cholmod as cm
from copy import copy

__all__ = ["Matrix", "MatMulChain"]


class Matrix:
    def __init__(
        self,
        matrix,
        *,
        invert=False,
        transpose=False,
        name=None,
    ):
        if isinstance(matrix, Matrix):
            self.matrix = matrix.matrix
            self.is_inverted = matrix.is_inverted
            self.is_transposed = matrix.is_transposed

            self._is_symmetric = matrix._is_symmetric
            self._is_diagonal = matrix._is_diagonal

            self.name = matrix.name
        else:
            if issparse(matrix) or isinstance(matrix, np.ndarray):
                assert len(matrix.shape) == 2
            elif isinstance(matrix, cm.Factor):
                pass
            else:
                assert False

            assert isinstance(invert, bool)
            assert isinstance(transpose, bool)

            self.matrix = matrix
            self.is_inverted = invert
            self.is_transposed = transpose

            self._is_symmetric = None
            self._is_diagonal = None

            self.name = name

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

    def __matmul__(self, other):
        if isinstance(other, (Matrix, MatMulChain)):
            return MatMulChain(self, other)
        else:
            assert isinstance(other, np.ndarray)
            assert self.shape[1] == other.shape[0]
            if len(other.shape) == 1:
                if self.is_inverted:
                    if self.is_diagonal:
                        return self.evaluate() @ other
                    elif self.is_factor:
                        raise NotImplementedError
                    elif self.is_sparse:
                        array = self.inv.evaluate()
                        return spsolve(array, other)
                    else:
                        array = self.inv.evaluate()
                        return np.linalg.solve(array, other)
                else:
                    return self.evaluate() @ other

            elif len(other.shape) == 2:
                if self.is_inverted:
                    if other.shape[1] >= other.shape[0]:
                        warnings.warn("inefficient solving")

                    if self.is_diagonal:
                        return self.evaluate() @ other
                    elif self.is_factor:
                        assert self.is_transposed
                        return self.matrix.apply_Pt(self.matrix.solve_Lt(other))
                    elif self.is_sparse:
                        array = self.inv.evaluate()
                        return spsolve(array, other)
                    else:
                        array = self.inv.evaluate()
                        return np.linalg.solve(array, other)
                else:
                    return self.evaluate() @ other

            else:
                assert False

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
    def is_symmetric(self, tol=np.finfo(float).eps):
        if self._is_symmetric is None:
            if self.is_factor:
                self._is_symmetric = False
            else:
                if self.shape[0] == self.shape[1]:
                    if self.is_sparse:
                        diff = abs(self.matrix - self.matrix.T) > tol
                        self._is_symmetric = diff.nnz == 0
                    else:
                        self._is_symmetric = np.all(self.matrix == self.matrix.T)
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
            new = Matrix(
                self.matrix,
                invert=self.is_inverted,
                transpose=not self.is_transposed,
                name=self.name,
            )
            return new

    @property
    def inv(self):
        new = Matrix(
            self.matrix,
            invert=not self.is_inverted,
            transpose=self.is_transposed,
            name=self.name,
        )
        return new

    def evaluate(self):
        if self.is_inverted:
            assert not self.is_factor

            if self.is_diagonal:
                diag = self.matrix.diagonal()
                inv = diags(1 / diag)
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

        if self.is_sparse:
            if self.is_diagonal:
                factor = diags(np.sqrt(self.matrix.diagonal()))
            else:
                factor = cm.cholesky(csc_matrix(self.matrix))
        else:
            factor = np.linalg.cholesky(self.matrix)

        sqrt_name = f"sqrt({self.name})"
        if self.is_inverted:
            return Matrix(factor, invert=True, transpose=True, name=sqrt_name)
        else:
            return Matrix(factor, invert=False, transpose=False, name=sqrt_name)

    def is_equal(self, other, tol=0):
        if self.is_factor != other.is_factor:
            return False
        elif self.is_inverted != other.is_inverted:
            return False
        elif self.is_transposed != other.is_transposed:
            return self.is_equal(other.T, tol=tol)
        elif self.shape != other.shape:
            return False
        else:
            if issparse(self.matrix):
                miss_count = (self.matrix != other.matrix).nnz
                if miss_count == 0:
                    return True
                elif miss_count <= tol:
                    warnings.warn("loose equality check")
                    return True
                else:
                    return False
            else:
                return np.all(self.matrix == other.matrix)


class MatMulChain:

    def __init__(self, *entries):
        self.chain = []
        self.scale = 1.0

        for entry in entries:
            self.append(entry)

    def __repr__(self):
        return " ".join([repr(matrix) for matrix in self])

    def __len__(self):
        return len(self.chain)

    def __getitem__(self, idx):
        item = self.chain[idx]
        if isinstance(item, list):
            return MatMulChain(*item)
        else:
            return item

    def __iter__(self):
        return iter(self.chain)

    def __next__(self):
        return next(self.chain)

    def __copy__(self):
        new = MatMulChain(self.scale, *self.chain)
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
        return MatMulChain(self.scale, *lst)

    @property
    def inv(self):
        lst = [matrix.inv for matrix in self[::-1]]
        return MatMulChain(1 / self.scale, *lst)

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

    def append(self, item):
        if np.isscalar(item):
            self.scale *= item
        elif isinstance(item, MatMulChain):
            self.scale *= item.scale
            for entry in item:
                self.append(entry)
        else:
            entry = Matrix(item)
            self.chain.append(entry)
        self.check_chain()

    def prepend(self, item):
        if np.isscalar(item):
            self.scale *= item
        elif isinstance(item, MatMulChain):
            self.scale *= item.scale
            for entry in item[::-1]:
                self.prepend(entry)
        else:
            entry = Matrix(item)
            self.chain.insert(0, entry)
        self.check_chain()

    def simplify(self, tol=0):
        skip_next = False

        for i in range(len(self) - 1, 0, -1):
            if skip_next:
                skip_next = False
                continue

            entry1, entry2 = self[i - 1], self[i]
            if entry1.is_equal(entry2.inv, tol=tol):
                self.chain.pop(i)
                self.chain.pop(i - 1)
                skip_next = True

        return self

    def evaluate(self):
        self.simplify()

        lst = [None] * len(self)
        for i, matrix in enumerate(self):
            if lst[i] is None:
                if matrix.is_inverted:
                    warnings.warn("explicit matrix inversion")
                    not_inv = matrix.inv.evaluate()
                    if matrix.is_sparse:
                        inv = np.linalg.inv(not_inv.todense())
                    else:
                        inv = np.linalg.inv(not_inv)
                    lst[i] = inv

                    # check for any identical inverses ahead
                    for j, upcoming in enumerate(self[i + 1 :]):
                        if upcoming == matrix:
                            lst[i + j + 1] = inv
                        elif upcoming.T == matrix:
                            lst[i + j + 1] = inv.T

                else:
                    lst[i] = matrix.evaluate()

        product = self.scale * eye_array(self.shape[0])
        for array in lst:
            product = product @ array

        return product

    def factorize(self):
        assert self.is_symmetric

        self.simplify()

        n = len(self)
        if n % 2 == 0:
            return MatMulChain(np.sqrt(self.scale), self[: n // 2])
        else:
            matrix_mid = self[(n - 1) // 2]
            factor = matrix_mid.factorize()
            return MatMulChain(np.sqrt(self.scale), self[: (n - 1) // 2], factor)

    def check_chain(self):
        for i in range(len(self)):
            shape = self[i].shape
            if len(shape) != 2:
                assert ValueError

            if i >= 1:
                if shape[0] != self.chain[i - 1].shape[1]:
                    assert ValueError
