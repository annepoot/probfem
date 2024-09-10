import numpy as np

from myjive.names import GlobNames as gn
from myjive.model.model import Model


class BoundaryObservationModel(Model):
    def GETOBSERVATIONS(self, globdat):
        Phi = self._get_phi(globdat)
        measurements = self._get_measurements(globdat)
        return Phi, measurements

    @Model.save_config
    def configure(self, globdat):
        pass

    def _get_measurements(self, globdat):
        fglobdat = globdat["fine"]
        dofs, vals = fglobdat[gn.CONSTRAINTS].get_constraints()
        return np.array(vals)

    def _get_phi(self, globdat):
        fglobdat = globdat["fine"]
        dofs, vals = fglobdat[gn.CONSTRAINTS].get_constraints()
        dof_count = fglobdat[gn.DOFSPACE].dof_count()

        Phi = np.zeros((dof_count, len(dofs)))

        for i, dof in enumerate(dofs):
            Phi[dof, i] = 1.0

        return Phi
