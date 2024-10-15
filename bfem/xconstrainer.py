import numpy as np
import scipy.sparse as spsp

__all__ = ["xConstrainer"]


class xConstrainer:
    def __init__(self, lhsconstraints, rhsconstraints, inputmatrix):
        self.update(lhsconstraints, rhsconstraints, inputmatrix)

    def get_lhs(self, u):
        uc = u.copy()
        dofs_lhs, vals_lhs = self._cons_lhs.get_constraints()
        uc[dofs_lhs] = vals_lhs

        return uc

    def get_rhs(self, f):
        fc = f.copy()
        fc += self._rhs
        _, vals_lhs = self._cons_lhs.get_constraints()
        dofs_rhs, _ = self._cons_rhs.get_constraints()
        fc[dofs_rhs] = self._coninv @ vals_lhs

        return fc

    def get_input_matrix(self):
        return self._input

    def get_output_matrix(self):
        return self._output

    def update(self, lhsconstraints, rhsconstraints, inputmatrix):
        self._cons_lhs = lhsconstraints
        self._cons_rhs = rhsconstraints
        self._input = inputmatrix
        self._output = spsp.csr_array(self._input.copy())
        self._rhs = np.zeros(self._output.shape[0])

        dofs_lhs, vals_lhs = self._cons_lhs.get_constraints()
        dofs_rhs, vals_rhs = self._cons_rhs.get_constraints()

        if not np.allclose(vals_lhs, 0.0) or not np.allclose(vals_rhs, 0.0):
            raise NotImplementedError("has not validated inhomogeneous bcs yet")

        self._rhs -= self._output[:, dofs_lhs] @ vals_lhs
        self._rhs[dofs_rhs] = vals_rhs

        conpart = self._output[np.ix_(dofs_rhs, dofs_lhs)].copy()
        self._output[:, dofs_lhs] *= 0.0
        self._output[dofs_rhs, :] *= 0.0
        self._output[np.ix_(dofs_rhs, dofs_lhs)] = conpart
        self._coninv = np.linalg.pinv(conpart.toarray())

    def constrain(self, k, f):
        assert k is self._input

        return self.get_output_matrix(), self.get_rhs(f)
