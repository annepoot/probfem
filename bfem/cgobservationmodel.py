import numpy as np

from myjive.names import GlobNames as gn
from myjive.model.model import Model
from myjive.util.proputils import split_off_type


class CGObservationModel(Model):
    def ASKMATRICES(self):
        return self._ask_matrices()

    def RETURNMATRICES(self, globdat):
        return self._return_matrices(globdat)

    def GETOBSERVATIONS(self, globdat):
        Phi = self._get_phi(globdat)
        return Phi, Phi.T @ globdat["extForce"], self._noise

    @Model.save_config
    def configure(
        self, globdat, *, matrix, renormalize, nobs, noise, solver={"type": "CG"}
    ):
        self._matrixname = matrix
        self._renormalize = renormalize

        solvertype, solverprops = split_off_type(solver)
        if solvertype != "CG":
            raise ValueError("Solver type must be CG!")
        self._solver = globdat[gn.SOLVERFACTORY].get_solver(solvertype, "solver")
        self._solver.configure(globdat, **solverprops)

        self._nobs = nobs
        self._noise = noise

    def _get_phi(self, globdat):
        u = np.zeros_like(globdat["state0"])
        f = globdat["extForce"]

        dofcount = globdat["dofSpace"].dof_count()
        if self._nobs is None:
            nobs = dofcount
        else:
            nobs = self._nobs

        PhiT = np.zeros((nobs, dofcount))
        PhiT[0] = self._solver.get_residual(u, f)

        for i in range(1, nobs):
            res = self._solver.get_residual(u, f)
            du = self._solver.iterate(res)

            u += du

            if self._renormalize:
                res = -self._solver.get_residual(u, f)
                mat = self._solver._matrix

                p = res
                for phi in PhiT[:i]:
                    p -= ((phi @ mat @ res) / (phi @ mat @ phi)) * phi

                assert np.allclose(self._solver._p, p)
                self._solver._p = p
            else:
                p = self._solver._p

            PhiT[i] = p

        return PhiT.T

    def _ask_matrices(self):
        needed = [self._matrixname]
        return needed

    def _return_matrices(self, globdat):
        if self._matrixname == "K":
            self._matrix = globdat[gn.MATRIX0]
        elif self._matrixname == "M":
            self._matrix = globdat[gn.MATRIX2]
        else:
            raise ValueError("Unknown matrix type!")
        c = globdat[gn.CONSTRAINTS]

        self._solver.update(self._matrix, c)
