import numpy as np

from myjive.names import GlobNames as gn
from myjive.model.model import Model


class BFEMObservationModel(Model):
    def GETOBSERVATIONS(self, globdat):
        Phi = self._get_phi(globdat)
        Phic = self._constrain_phi(Phi, globdat)
        return Phic, Phic.T @ globdat["fine"]["extForce"], self._noise

    @Model.save_config
    def configure(self, globdat, noise):
        self._noise = noise

    def _get_phi(self, globdat):
        cglobdat = globdat["coarse"]
        fglobdat = globdat["fine"]

        elemsc = cglobdat[gn.ESET]
        nodesc = cglobdat[gn.NSET]
        nodes = fglobdat[gn.NSET]
        dofsc = cglobdat[gn.DOFSPACE]
        dofs = fglobdat[gn.DOFSPACE]

        rank = fglobdat[gn.MESHRANK]
        shapefac = fglobdat[gn.SHAPEFACTORY]
        shape = shapefac.get_shape(fglobdat[gn.MESHSHAPE], "Gauss1")
        dof_types = fglobdat[gn.DOFSPACE].get_types()

        Phi = np.zeros((dofs.dof_count(), dofsc.dof_count()))

        # Go over the coarse mesh
        for inodesc in elemsc:
            coordsc = nodesc.get_some_coords(inodesc)

            # Get the bounding box of the coarse element
            bbox = np.zeros((rank, 2))
            for i in range(rank):
                bbox[i, 0] = min(coordsc[:, i])
                bbox[i, 1] = max(coordsc[:, i])

            # Go over the fine mesh
            for inode, coords in enumerate(nodes):

                # Check if the node falls inside the bounding box
                inside = True
                for i in range(rank):
                    if coords[i] < bbox[i, 0] or coords[i] > bbox[i, 1]:
                        inside = False
                        break

                # If so, check if the node falls inside the shape itself
                if inside:
                    loc_point = shape.get_local_point(coords, coordsc)
                    inside = shape.contains_local_point(loc_point, tol=1e-8)

                # If so, add the relative shape function values to the Phi matrix
                if inside:
                    svals = np.round(shape.eval_shape_functions(loc_point), 12)
                    idofs = dofs.get_dofs([inode], dof_types)

                    for i, inodec in enumerate(inodesc):
                        sval = svals[i]
                        idofsc = dofsc.get_dofs([inodec], dof_types)
                        Phi[idofs, idofsc] = sval

        return Phi

    def _constrain_phi(self, Phi, globdat):
        Phic = Phi.copy()
        cdofs, _ = globdat["coarse"][gn.CONSTRAINTS].get_constraints()

        for i in range(Phic.shape[0]):
            for cdof in cdofs:
                if np.isclose(Phic[i, cdof], 1):
                    Phic[i, :] = 0.0
                    Phic[:, cdof] = 0.0

        return Phic
