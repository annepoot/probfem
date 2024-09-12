import numpy as np

from myjive.names import GlobNames as gn
from myjive.model.model import Model
from myjive.util.proputils import split_off_type


class RandomObservationModel(Model):
    def GETOBSERVATIONS(self, globdat):
        Phi = self._get_phi(globdat)
        return Phi, Phi.T @ globdat["fine"]["extForce"], self._noise

    @Model.save_config
    def configure(self, globdat, *, renormalize, nobs, seed, noise):
        self._renormalize = renormalize
        self._nobs = nobs
        self._seed = seed
        self._noise = noise

    def _get_measurements(self, globdat):
        return self._Phi @ globdat["fine"]["extForce"]

    def _get_phi(self, globdat):
        dofcount = globdat["fine"]["dofSpace"].dof_count()
        if self._nobs is None:
            nobs = dofcount
        else:
            nobs = self._nobs

        PhiT = np.zeros((nobs, dofcount))
        rng = np.random.default_rng(self._seed)

        for i in range(nobs):

            p = rng.standard_normal(dofcount)

            if self._renormalize:
                newp = p
                for phi in PhiT[:i]:
                    newp -= ((phi @ p) / (phi @ phi)) * phi
                p = newp

            PhiT[i] = p

        return PhiT.T
